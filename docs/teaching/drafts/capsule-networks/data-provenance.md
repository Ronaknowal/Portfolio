# Capsule Networks: data and author-calculation provenance

Prepared 13 September 2026, content-only revision 1. These offline inputs, programs and outputs are necessary pending implementation artifacts, not disposable scratch. No website lab, browser rendering, GPU benchmark, full MNIST reproduction or large-scale capsule experiment was implemented.

## Real licensed input

The source is **Optical Recognition of Handwritten Digits**, E. Alpaydin and C. Kaynak (1998), [UCI dataset record](https://archive.ics.uci.edu/dataset/80/optical+recognition+of+handwritten+digits), [DOI 10.24432/C50P49](https://doi.org/10.24432/C50P49). The UCI record, opened during this research pass, specifies CC BY 4.0. Preserve attribution, license and transformation notice.

digits-400.csv is an unchanged copy of the local attributed subset used in Landmark/Depthwise/ConvNeXt packets: first 40 occurrences of every class 0–9 from sklearn's 1,797-row load_digits source, concatenated by class. This source derives from the optical-digits test file; the teaching subdivision is not the original UCI benchmark split and is not MNIST. source_id is the one-based original row; pixel_0 through pixel_63 are row-major 8×8 intensities 0–16; digit is the class. CSV SHA-256 **a5b50ff0418e2b470140153a399c9200b2bba68468232507fd393ba89c4fc672**, 61,444 bytes.

Fresh executed pre-fit checks found 400 distinct source IDs and 400 distinct complete pixel vectors. No writer/session IDs are available; exact duplicate checks do not establish independent writers or absence of related handwriting. Fixed scaling divides by the published maximum 16 without fitting statistics.

A stratified train_test_split(test_size=.3, random_state=22) gives 280 train and 120 development images, 28/12 per class. All exact IDs and development labels are saved in calculated-inputs.json. No development rows enter fitting. Comparisons consume the development information; no untouched final-test score is claimed.

## Executed complete learning program

capsule-learning.py ran under Python 3.12.14, NumPy 2.3.5, PyTorch 2.14.0+cpu and sklearn 1.9.1, one CPU thread. Dependencies were already installed in the shared read-only runtime; nothing was installed or mutated there.

Six runs, seeds 1/2/3 × routing steps 1/3. Same seed produces identical initial common weights and decoder, since routing count changes no parameters or initialization draws. Batch generator seed100+seed pairs the 600 batches of 64 training draws with replacement. All fits use Adam(lr=.003), default remaining optimizer options, no weight decay, no augmentation, no early stopping and no checkpoint search. There is no fitted dropout or batch normalization to confound train/eval behavior. Model weights are not shared between fits after initialization.

Exact model:
- Conv 1→32, 3×3, padding1, bias, ReLU.
- Primary Conv32→16, 3×3, stride2, padding1, bias.
- Regroup B,type4,coordinate4,H4,W4; permute to B,H,W,type,coordinate; flatten 64 children, 4 coordinates each; squash.
- Vote transforms64×10×8×4, initialized normal std.1. Other layers use native defaults.
- Routing: parent-axis softmax, weighted sum over children, squash, dot-product agreement and accumulated logits; logits initialize zero every call. All routing steps are differentiated during fitting.
- Class score = capsule Euclidean length. Argmax resolves ties toward the lowest class index.
- Decoder: masked80→64→128→64 with ReLU/ReLU/sigmoid. True-class mask during training; predicted-class mask for ordinary evaluation; true-class-conditioned evaluation retained separately.
- Objective = batch-mean sum of margin terms (.9/.1/.5) + .0005 × batch-mean pixel SSE (64 pixels). No cross-entropy/probability reinterpretation.

Each model has 47,184 learned parameters. Metrics recorded at updates0,1,100,300,600. All final training correct counts are280. Reconstruction baseline is a training-mean image, development MSE .07295990735292435. Exact final metrics:

| Seed | Training/evaluation routing | Correct/120 | Margin loss | Predicted-mask MSE | True-label-conditioned MSE |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 1 | 117 | .04679877683520317 | .03125938028097153 | .030409611761569977 |
| 1 | 3 | 117 | .023643914610147476 | .03333989158272743 | .03293294832110405 |
| 2 | 1 | 117 | .04823140799999237 | .031032979488372803 | .030518144369125366 |
| 2 | 3 | 116 | .033108849078416824 | .03400780260562897 | .03358541429042816 |
| 3 | 1 | 117 | .046080294996500015 | .03324747085571289 | .03276568651199341 |
| 3 | 3 | 118 | .02497371844947338 | .03552311286330223 | .035221900790929794 |

Fixed-weight inference interventions at counts1/2/3/5 are stored for all six fitted models, including all lengths, labels, predictions and reconstruction metrics. Rows in the table are not additional fits. Correct counts alone are not prediction identity; author-check-results.json records the number changed from each model's training-time inference count. The seed3 trained1 model changes one prediction at inference2 while retaining117 correct. Full reconstructions are retained only for the selected examples/edits that need them; redundant per-image arrays for every intervention were removed after preserving the metrics and predictions. Compact JSON formatting preserves every retained number without rounding.

Nonwrapping shifts (row,column) = (0,1),(1,0),(0,0) are computed from real development images, with zero fill and truncation at the boundary. They are distribution-shift probes with an explicit possible label-preservation failure, not formal equivariance tests or adversarial robustness certificates. Zero-shift correct counts match the unmodified inputs. All outcomes, including substantially worse shifted performance, are retained. No fit was restarted or selected to make a preferred model win.

The complete seed1 models for both routing configurations are saved, along with the first two development images (source251, actual4; source40, actual9), class vectors and all routing trace states. This deterministic selection precedes examining their error status. The two sets of trained weights are distinct; callers must bind predictions to the chosen model.

## Constructed mathematical fixtures

capsule-mechanics.py and mechanics-results.json contain author-generated arithmetic examples, explicitly not empirical measurements:
- Three children/two parents/2D votes; full8-step trace; opposing vote edit; zeros; identical parents; child permutation.
- Independent NumPy scalar-child/parent routing agrees exactly in float64 with the Torch route on16 combinations (four fixtures ×1/2/3/8 steps).
- Squash values/Jacobians at [.3,.4], [0,0], [3,4], analytical radial/tangent eigenvalues and central differences.
- Full3-step objective half squared distance to target[[.2,-.1],[.5,.3]], full gradient versus central differences max error1.15464e−10. The chosen stop-gradient path yields the same loss .5170857167403664 but max difference .03751796471 against the derivative of the full forward function.
- Orthogonal rotation/squash commutation, scaling counterexample, unconstrained linear-map/rotation failure and explicit homogeneous pose relation. These are supplied coordinate systems, not learned-pose discoveries.
- Three-round diagonal EM illustration with votes A[0,.2,2], B[0,3,3.2] in coordinate1 and zeros in coordinate2, activations[1,1,.5], beta_u=beta_a=0, variancefloor.01, inverse temperatures .5/.75/1 and massguard1e−12. Responsibilities are normalized per child; weights=a*R are recomputed per step. Exact mass, means, variances, activations and responsibilities retained. Inactive-third-child vote edits leave means unchanged. This illustrates the declared equations, not trained EM-CapsNet.
- Exact classic vector-CapsNet shape/parameter and forward arithmetic counts. Total8,215,568 parameters; vote tensor at B32 consumes23,592,960 bytes in float32. Counts omit optimizer/backward/softmax/squash costs when stated; no time benchmark is inferred.

The small capsule_learning_import.py helper imports the hyphenated downloadable program without running its main, so mechanics/author checks never refit models. All three scripts and helper are complete and executable with the documented dependencies.

## Independent selected-example arithmetic and edit inputs

author-checks.py reads saved state, reconstructs the two convolutions using explicit spatial loops and NumPy contractions, independently maps every primary location/type/coordinate, forms each vote by matrix multiplication, uses the NumPy routing implementation and decodes via explicit matrix products/sigmoid. Four comparisons: two models × two saved specimens.

Maximum difference from saved Torch float32 capsule coordinates: **6.385632755900872e−8**. Maximum reconstruction difference: **2.977743263077315e−7**. This is author arithmetic evidence, not independent external review or browser parity.

For every comparison, flip real pixel(row3,column3) to1−old and retain new lengths, winner and capsule changes. Also decode coordinate0 edits of the selected predicted class at delta−.1/0/+.1; retain actual reconstructed arrays and maximum changes. Editing an unselected class by.1 leaves reconstruction exactly unchanged. The model does not use an annotation label unless it is explicitly selected as the decoder mask.

## Limits and continuation

All displayed author programs were executed, except the shortened illustrative route snippet, whose operations are the exercised default path of the complete program. Package-install commands document the tested dependency versions but were not executed in this pass because dependencies already existed. Full MNIST/affNIST/MultiMNIST, EM-CapsNet, variational routing, STAR-Caps and learned geometric-pose experiments were not run. External paper results are cited historical findings, not this packet's measurements.

calculated-inputs.json retains wider author evidence; phase two should extract only the necessary selected model/input/figure records into lazy topic-owned assets. Do not eagerly ship all experiment JSON or retrain the six models in a browser. Preserve exact source IDs, model identity, coordinate ordering and score semantics when packaging.

No native large-model reproduction, rendering, JavaScript/worker implementation, accessibility/browser performance campaign, formal phase-two correctness review or publication was performed. Existing runtime source remains unchanged. Keep these packet artifacts and clean only disposable author-owned cache after verifying exact paths.
