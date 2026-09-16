# Convolution: visual and investigation specifications

Research/write deliverable only. No React, SVG, calculation engine, download integration or browser investigation has been implemented. Use the retained convolution-experiments.py, calculated-inputs.json and attributed CSV as inputs. These homes serve distinct mechanisms; preserve their spatial correspondence instead of replacing them with repeated form-and-output boxes.

## Shared interaction and accuracy contract

Every scored prediction is initially unset, requires submission before revealing its answer, and binds to the actual selected values, geometry, target, model/run/step and content revision relevant to that question. Grade the stated numeric quantity or relation from those inputs, not a canned “correct” button. Unless noted, exact tiny arithmetic uses tolerance 1e−8; saved float32 model comparisons use stored numbers and a declared 1e−7 tie threshold. Changing relevant input clears the answer, revealed result and success. Reset restores the stated initial unsolved fixture and clears progress within the activity. Preset selection alone does not count as construction.

Every diagram has a readable equation/row-column table equivalent. Coordinates and labels carry meaning independently of color; signed quantities use zero-centered scales with visible legends. Keyboard arrows/direct numeric input duplicate dragging; no required hover or animation. Stack matrices with repeated axis labels on narrow screens. Feedback uses a polite live region, retains focus, and names the tested quantity. Respect reduced motion. Formula errors or impossible dimensions produce explanatory states without NaN or off-canvas output.

Small editable arrays are bounded below. Real training/gradient maps are selected from recorded cases, not recalculated by loading torch into the browser. Load only the current lesson and visible heavy data home as appropriate; don't preload all twelve fit/map records globally. Phase two must check translated calculations against exact and contrast fixtures, keyboard/mobile/text alternatives, grading/reset/invalidation, dimensions/units, lazy loading and actual payload size.

## 1. Patch, products and affected outputs

Home: §1, fixtures.patch. Three separate but linked grids: 3×3 image, 2×2 filter, 2×2 output. Window coordinates remain visible. At each location show four products entering the sum. Source input [[1,2,0],[0,1,3],[2,1,0]], weights [[1,−1],[0,1]], no bias, unit stride, no padding. Outputs [[0,5],[0,−2]].

Initial prediction: selected top-right output, numeric blank; grade 5 before reveal. Window movement invalidates the prediction. Edit image values in [−5,5] and weights [−3,3], steps .5; render the current selected sum, not cached source output.

Construction: begin with central image value 1. Task: change just that cell to make bottom-right output −1 while preserving top-right output 5. Current bottom-right is −2, so initial state fails. Central value 2 succeeds; value 3 gives bottom-right 0 and fails target. Check all constrained cells. The initial top-left output changes 0→1 under this edit; do not imply only one output changed.

Null: zero filter makes all outputs zero for any image (fixtures.nulls). A different contrast uses diagonal filter [[1,0],[0,−1]]: center edit changes only top-left/bottom-right because the other appearances have zero weight. These are different causes of unchanged outputs, not disconnection inferred from one zero.

Optional fixed-derivative inset: [1,0,−1] on five ones gives three zeros without padding. One impulse contrasts with the constant case. Label it a hand-specified signal operator.

## 2. Channel stack and connection construction

Home: §2. Two 1×1 input channels x=[2,3], output weight row [4,−1], bias1. Animate two partial products into output6; a second output row uses its own weights. An axis diagram maps [N,C,H,W] and [Cout,Cin,kh,kw] without assuming a colored map is an RGB channel.

Prediction asks which numeric output follows the selected row. Changes of channel values, row or bias invalidate the answer.

Construction: start two output rows [[1,1],[1,1]], bias[0,0], task sum and difference for x=[2,3], contrasting x=[−1,4], and x=[0,0]. First row initially works; second fails (5 instead of −1). Learner edits four weights [−2,2] and two biases [−2,2]. Grade all three inputs: desired [5,−1], [3,−5], [0,0]. The zero input checks the biases; the other two independent vectors determine the two coefficients. A solution [[1,1],[1,−1]], zero bias, works. To establish channel isolation instead, a separate unscored demonstration fixes [1,0], changes second input, and shows unchanged first result; zeroing all weights gives a null that must not pass the sum/difference task.

Groups inset: four inputs→six outputs with groups2; show each three-output block connected to its two-input block. Mark divisibility errors before computing. Depthwise multiplier2 uses two inputs→four outputs; fixtures.parity contains those actual shapes. No full advanced architecture lab required here.

## 3. Shared weight and gradient accumulation

Home: §3, fixtures.shared_update. Two windows [1,3] and [3,2] feed the same weight pair [1,−1], targets[0,0], half-SUM error. Output[−2,1], loss2.5, weight gradient[1,−4], input gradient[−2,3,−1]. Show location contributions separately, then their sum at each shared parameter.

Initial prediction blank: new first weight after learning rate .1; answer .9. After submission, reveal full new weights[.9,−.6], outputs[−.9,1.5], loss1.53. Explain that improving the total does not imply each output error improves.

Contrast: change target second coordinate to2. Output gradient becomes[−2,−1], weight gradient[−5,−8]; updated weights[1.5,−.2], output[.9,4.1], new loss2.61 versus initial2.5—this particular .1 step overshoots. Learner lowers the rate; .01 gives weights[1.05,−.92], outputs[−1.71,1.31], loss1.7001. Author arithmetic retained via the fixture additions; do not silently keep a “descent always improves” caption.

Null target equal to current outputs gives zero loss/gradients/update. Free edit targets [−3,3], learning rate [0,.2]; show current target and reduction. Prediction revision includes target and rate. The zero-rate case shows unchanged weights even with nonzero gradient.

## 4. Window geometry and alignment repair

Home: §4, fixtures.shape_cases. One-dimensional ruler before a two-dimensional extension. Explicitly label real/padded positions, sampled kernel dots, next start and leftover unused positions. Output count follows floor((n+left+right−d(k−1)−1)/s)+1. First center .5+d(k−1)/2−left.

Initial n8,k4,s1,d1,left1,right1 is unsolved for target output8. Prediction blank asks its actual output size7; submit before reveal. Construction asks learner to achieve output8 with total padding3, each side≤2, while matching requested first center1. Only left1/right2 matches both; left2/right1 has output8 but center0, a useful rejected contrast. Left0/right3 is excluded by the stated bound. Null k1,s1,p0 is identity geometry, but weight values still determine the operator.

Explore n[3,16],k[1,5],s[1,3],d[1,3],padding each[0,4]. Reject nonpositive outputs. Show “same” as a policy for this stated stride-one condition; no implication that PyTorch same permits arbitrary stride. Source fixtures include n7,k3,s2,no padding→3 and dilation-two preserved shape.

Boundary-mode comparison uses fixed values with zero, circular, reflection or replicate semantics only if implemented with correct respective constraints. It is an optional extension, not four obligatory modes; circular versus zero fixture in home9 is the required contrast.

## 5. Pool routes and reconstructability

Home: §5, fixtures.pooling. Side-by-side max winner paths and average contribution paths share an editable small input. For [1,4,3], k2,s1, output [4,4], gradient of output sum [0,2,0]. Unset prediction asks middle gradient; grade2. Contrast [3,1,4] yields[3,4] and[1,0,1].

Construction: two editable 2×2 windows, initial both zero. Require different arrays, maximum4 and average2 in each, with entries[0,4]. Examples [[4,2],[1,1]] and[[4,0],[2,2]] pass; duplicating the first fails the “different windows” constraint even if summaries match. An all-zero pair is a checked null for both aggregate calculations and must fail target. Display why this proves nonunique reconstruction.

Within-window versus boundary motion uses [0,1,0,0]→[1,0,0,0] (pool output[1,0] unchanged) versus [0,0,1,0]→[0,1]. Ties [2,2,1] use actual CPU routing but distinguish implementation choice from a unique derivative.

Padding inset is a compact diagram, not a new lab: negative input[−2,−3], max pad−infinity→[−2,−2]; mean including zero padding−5/3 versus excluding−5/2. Adaptive five→three bins show overlapping colored brackets and outputs1.5/3/4.5. Sum-of-output gradient [.5,5/6,1/3,5/6,.5] exposes the overlap. Bin formula start=floor(i*n/m),end=ceil((i+1)*n/m), output i0..m−1.

## 6. Coordinate ancestry and a sparse field

Home: §6, fixtures.receptive_trace and dilation_offsets. Layer strips show output width, r,j,a together. Starting32 follows conv3/pool2/conv3/pool2/conv3/global-average8. Required numbers: r3/4/8/10/18/46, j1/2/2/4/4/4, centers.5/1/1/2/2/16.

Prediction blank asks r after second pooling; grade10. Optional selected output then shows center=a+index*j and traced discrete ancestors. Padding is hatched outside input; bounding width and unique observed coordinate count are separate labels.

Construction: two layers initially both k3,d2,s1 with matching symmetric padding. Task use two k3 filters, unit strides, dilation choices1/2, to reach all seven offsets from−3..3 using a seven-wide bound. Initial d2/d2 reaches only five offsets spanning9 and fails. d1/d2 or d2/d1 passes; d1/d1 reaches5 and fails. Check actual offset set, not just r. Null k1/k1 has r1 and cannot pass.

Branch overlay uses equal output dimensions but different a to illustrate misalignment; grade no new merge task unless the implemented controls really edit both paths. The source's deeper practice stride2→dilation2 has nine connected offsets in an11-wide bound with holes±2; retain as changed transfer.

## 7. From digit pixels to measured evidence

Home: §7. Draw the exact shape strip and then actual saved seed1 specimen maps for cnn_max/cnn_average/cnn_global_average. Source IDs251/40/149, labels4/9/8, all predicted correctly by these saved models. Only these cases have saved maps; don't fabricate maps for arbitrary seed or stress-shift images.

A specimen viewer selects current input, first-layer filter0..7 and its 8×8 responses; second maps have16 channels/4×4; final maps16/2×2; probability bars have10 classes. Raw logits are not saved. Numbers and signed filter scale remain visible; first convolution raw preactivation isn't saved, so reconstructing it requires the bias, which is not saved. **Show saved post-ReLU maps, explicitly labeled; do not pretend to derive an exact raw response from kernel alone without its bias.** Use recorded first_layer_kernels as a separate weight view.

Evidence prediction: choose lower final validation CE between two selected measured models, seed initially1, dense versus max-CNN. Submit lower/equal/higher before reveal; dense.07949448 versus max.08332337. Seed3 reverses direction. Reset seed1 pair without answer. Different step selection binds a fresh prediction to actual step0/1/25/100/200/400; plot only those measured points and label connecting segments as interpolation.

A separate stress table shows all recorded right/down zero-fill shifts and unchanged label convention, with clipping caveat. No unrecorded predictions per shifted specimen. Explain changed protocol versus tensor equivariance. Null same model/seed/step comparison must grade equal. Counts always show denominators; CE is mean over rows. Construction is the written changed-training protocol practice, not a misleading browser “train” button.

## 8. Theoretical reach versus actual influence

Home: §8, fixtures.linear_profiles; recorded fit visual_rows.logit_input_gradient. Two explicitly different panels.

Exact linear panel repeatedly convolves [1,1,1]/3 with no nonlinearities. Depth options2/3/5/10/20; support widths5/7/11/21/41. Depth2 profile[1,2,3,2,1]/9; depth20 1%-peak width21. Draw actual coefficients at integer offsets; Gaussian overlay only if labeled analytic approximation with stated variance2L/3, never substituted data.

Prediction: before reveal, enter the width that remains at the selected depth and relative-to-peak threshold. Initial depth20/threshold1% grades21; depth10/1% grades15. General threshold in[.001,.2] recomputes coordinates satisfying value≥threshold*peak. Answer binds depth and threshold. Discuss differing support/concentration growth from those measured widths separately; do not ask a fixed yes/no question whose answer cannot change across the allowed settings.

Empirical panel shows signed gradient of the recorded predicted logit at the saved input with model/seed1/source ID/class clearly bound. It is local sensitivity, not causal attribution or literal gaze. Selecting another specimen selects real saved input. Displaying a different identity label alone must not alter numbers.

Null dead-ReLU fixture gives all zero gradients despite architectural support. Never divide by zero in heatmap normalization; show an explicit zero state. The positive linear profile cannot demonstrate signed cancellation, so retain prose qualification. Any future interactive weight-editing ERF needs a new complete computation spec; do not imply saved-map selection is editing a trained model.

## 9. Shift and sample phase

Home: §8, fixtures.translation. A six-position cyclic signal and stride-one derivative show input shift, computed output shift and difference. Initial one impulse, kernel[1,0,−1]. Unset prediction asks max difference for circular boundary:0; changing boundary to zero yields1 and clears the answer.

Sampling activity shows alternating±1 sampled at stride2. Initial phase0 gives[1,1,1]; phase1 gives[−1,−1,−1]. Learner must construct a nonnegative two-tap filter whose weights sum1 and suppress this alternating pattern for both phases. Start weights[1,0], unsolved; [.5,.5] passes with exact zero. [.75,.25] gives opposite±.5 and fails. All-zero weights are a checked null for output but fail the sum1 constraint. Bounds[0,1], numeric input; grade both phases plus weight sum.

Do not claim that this single-frequency solution guarantees general shift invariance. Circular extension avoids changing signal mass here; the real digit stress test uses zero fill and clipping and belongs to a different evidence panel.

## 10. Transpose as overlapping contributions

Home: §9, fixtures.adjoint and transposed_coverage. Matrix view pairs C rows with local windows; scatter view sends each dual input into C-transpose columns. Initial x[1,3,2],g[2,−3], inner products−7, transpose[2,−5,3]. Prediction blank asks middle transpose coordinate−5. Null g0 yields0 but does not establish invertibility; C-transpose Cx[−2,3,−1] contrasts withx.

Coverage explorer input three ones,k3,s2 gives[1,1,2,1,2,1,1]. Switch k4 gives[1,1,2,2,2,2,1,1]; distinguish equal interior overlap from boundaries. Two-dimensional coverage is outer product of 1D coverage for this all-ones separable example, values1/2/4. Numerical contributions on each cell accompany color.

Optional exploration: choose k in{2,3,4} at s2 and compare interior coverage;3 varies,2/4 are constant with explicit interior exclusion defined from complete footprints. This preset comparison is not counted as a construction exercise. No claim that trained artifacts disappear; learned weights need their own assessment. Scalar output-size panel uses transposed formula and separates output_padding from painted zero cells.

## 11. Patch matrix, overlap counts and operation budget

Home: §10. Static/explorable correspondence is sufficient: each unfolded column matches one outlined input patch and flattening order. Matrix-vector multiplication yields[0,5,0,−2]. Folding produces[[1,4,0],[0,4,6],[2,2,0]], counts[[1,2,1],[2,4,2],[1,2,1]]. Dividing restores input; label zero-count holes unrecoverable if an extended stride creates them.

Budget table is formula-driven, not a performance graph: first/second/head parameter counts80/1168/650; MACs4608/18432/640. MLP2368 MACs/2410 parameters. Annotate excluded activations/pooling and two-FLOPs-per-MAC convention. Inputs may edit output shape/channels only with valid groups and matching graph. Do not turn parameter ratio into speedup.

Inference-fold inset traces BN scale into each filter and bias using stored formula/float64 parity error3.33e−16. ReLU remains a separate mathematical operation. Local fixed-stencil application shows center30/four neighbors20→−40, then units1/h² and diffusion coefficient separately; no time evolution without a defined stable discretization and boundary protocol.

## Deferred implementation review

Translate only bounded toy formulas and preserve actual recorded model cases; initial maps/labs still require accessible code-native implementation. Verify any added dynamic controls with changed and null inputs, float precision/threshold semantics, ties and invalid geometry. Check downloadable code/CSV paths, canonical next-topic link, source/resource annotations, mobile readability, color contrast, keyboard-only task completion and reduced-motion behavior. Run relevant app/content/manifest/curriculum/build checks only during phase two. This document is not evidence that those have passed.
