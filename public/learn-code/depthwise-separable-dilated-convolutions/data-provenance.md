# Data, programs and calculation provenance

Prepared 13 September 2026 for research and writing only. Keep the entire packet until its later implementation is complete; its data and numerical outputs are required inputs, not temporary screenshots.

## Actual data and declared split

Source: **Optical Recognition of Handwritten Digits**, E. Alpaydin and C. Kaynak (1998), [UCI dataset record](https://archive.ics.uci.edu/dataset/80/optical+recognition+of+handwritten+digits), [DOI10.24432/C50P49](https://doi.org/10.24432/C50P49). The UCI record inspected on 13 September 2026 distributes it under **CC BY4.0**. Preserve this attribution, license and transformation notice. These are optical digits, not MNIST.

The packet's digits-400.csv is an unchanged copy of the attributed subset in the preceding Landmark/Convolution packets: first40 occurrences of each label0–9 in scikit-learn's1,797-row load_digits copy, concatenated by label. source_id is the one-based source row; pixel_0…pixel_63 are row-major8×8 integer values0–16, digit the label. This sklearn copy originates from the original UCI test file, but our subdivision is a separate teaching investigation, not that benchmark's train/test protocol.

CSV SHA-256 a5b50ff0418e2b470140153a399c9200b2bba68468232507fd393ba89c4fc672;61,444bytes. This real source was selected because its inputs are small enough to inspect every pixel, train bounded CPU models and store complete model weights for interactive inference. Reusing it permits a related question—post-training filter factorization—without implying architectural rankings are directly comparable with earlier different-model experiments.

Fresh pre-fit inspection in convolution-factorization.py verified400 unique sourceIDs and400 distinct complete pixel vectors. No exact image duplicate crosses the split. Writer IDs are unavailable; distinct vectors do not guarantee independent writers or exclude related specimens. No independent-writer claim is made.

Normalize by the published fixed intensity limit16, with no fitted preprocessing. Split indices using stratified train_test_split(test_size=.3,random_state=22):280 training,120 development,28/12 per class. Complete ID arrays are in calculated-inputs.json. Development drives comparative interpretation and is consumed; there is no untouched final test, post-selection deployment estimate or original-benchmark comparison.

## Actual six fits and 24 fixed factorizations

convolution-factorization.py executed on Python3.12.14, NumPy2.3.5, PyTorch2.14.0+cpu, scikit-learn1.9.1, one PyTorchCPUthread. No dependency installs, pretrained downloads, CUDA, heavyweight architecture reproduction or timing benchmark.

Model: Conv1→8 k3p1/ReLU; Conv8→12 k3 with d1p1 or d2p2/ReLU; averagepool2 stride2; flatten192→linear10. Input N×1×8×8, intermediate N×8×8×8 then N×12×8×8, pooledN×12×4×4. All bias terms retained. Within each seed1,2,3 both dilation models receive identical initial tensors because the module construction and seed are identical; the spacing/padding choice changes the forward function. Each is trained separately with Adam.003 for400 full-batch steps; no augmentation, normalization, weight decay, dropout, early stopping or checkpoint selection.

At steps0,1,25,100,200,400 score training and development with eval/no_grad, recording correct/count, cross-entropy, predictions and per-class correct. Repeated score calls do not update parameters. All six final training counts280/280. Six fits, not24 further fits: multipliers1/2/4/9 are constructed from fixed trained second-layer weights without retraining or using development to calculate factors.

For each original input channel, reshape its filters to12×9 and compute reduced SVD. Put retained Vh rows into m depthwise3×3 filters, U columns multiplied by retained singular values into a pointwise12×(8m) kernel. Original spatial bias becomes pointwise bias; depthwise is unbiased. No activation between factors, original ReLU follows the pair. SVD gives optimal rank-m Frobenius weight approximation for each input-channel matrix, not optimal task loss. A singular vector sign change paired in U/Vh leaves reconstructed weights unchanged.

All-model parameters/MACs: dense2886/61824; m1 2190/17280; m2 2358/28032; m4 2694/49536; m9 3534/103296. MACs count convolution/linear products accumulated per image, excluding bias, activation, pooling, memory and other operations. Costs agree for both dilations because padding preserves output shape.

Final development correct out of120 in order dense,m1,m2,m4,m9:

| Seed/dilation | Counts | Dense CE |
| --- | --- | --- |
|1/1|117,112,115,117,117|.1286953092|
|1/2|115,95,110,113,115|.1459311545|
|2/1|116,90,112,116,116|.1829237044|
|2/2|116,56,96,112,116|.1279900670|
|3/1|115,107,114,115,115|.1697738022|
|3/2|115,98,106,114,115|.1379346102|

calculated-inputs.json preserves every factorization's CE, relative weight error and maximum absolute logit difference across all development observations. Example seed1d1 m4 CE.137114 versus dense.128695 despite equal117counts. Worst rank9 max-absolute-logit change across six runs1.71661376953125e−5. Largest m1 damage seed2d2:56correct, CE2.253332. No fine-tuning recovery curve was generated. Seeds express training variability on one split, not independent dataset replications.

## Retained model state and actual/constructed inputs

Both seed1 dense models retain full state_dict values. Each of their four factorized spatial layers also retains full weights/bias. Other seeds retain metrics and predictions, not editable full models. Complete seed1 model states permit deterministic small forward computation on edited8×8 inputs during phase two; no browser SVD or fitting is required.

Selected real inputs: first development observation source251/actual4 for each dilation, plus first disagreement between dense and m1: source299/actual1 for d1, source32/actual9 for d2. The latter are deliberately selected failure/contrast examples; not random samples or population-frequency evidence.

author-checks.py independently reconstructs stem, spatial convolution/factors, ReLU, average pooling and head using NumPy plus explicit scalar convolution loops. Across four saved real inputs, max dense-loop versus stored PyTorchfloat32 logit difference1.1512556547188524e−5. All16 saved factorization predictions match the corresponding recorded development argmax. Author-loop factor logits are retained.

Constructed pixel intervention: replace normalized cell[0,0] from0 to1. Saved actual-class logit deltas: source251d1−.8635377838, source299d1−1.4089149643, source251d2−1.4188996912, source32d2+1.1825324818. All edited logits/argmaxes retained in author-check-results.json. This is a corruption experiment on real inputs, not a new labeled dataset; original labels need not describe arbitrary edited images. Restoring the pixel is a deterministic null.

## Exact fixtures and operator checks

author-checks.py's direct-loop cross_correlation covers groups, zero padding, integer stride/dilation and dtype-preserving outputs. Compared against Torchfloat64 for groups1/2/4 and (d,s)=(1,1),(2,1),(2,2) on seeded4×5×6 input and8output kernels. Nine comparisons, largest error5.329070518200751e−15. Small finite probes substantiate these outputs; they are not a blanket all-device implementation certification.

Other retained calculations:

- Two-channel two-tap forward and chain-rule gradients; η.01 output−1.9824/loss4.44735488 from initial−3/loss8. Independent hand substitution agrees.
- Rank-two identity versus explicit rank-one projection, changed patch[2,3] and null[2,0].
- Nine-cell d1/d2/d8 stencil, touched and untouched-cell edits, exact serial support sets for six schedules and8×8 finite-map valid-tap grids.
- Dense/separable weights/MACs and one-output counterexample; MBConv stride-one versus stride-two area placement.
- Parallel context constructed vector[center,d1sum,d2sum,d4sum,mean], projection[0,.1,.2,.3,1]: base14, index8→20 gives18.5222222222, swapping indices0/1 preserves globalmean5 but changes local projection14→14.3.
- Changed practice budgets, receptive-field jump, hard-swish values and four-layer structural81-site bound.

The complete inline linear-pair program from lesson section4 was separately executed: shapes(1,6,7,7) and(1,5,7,7), allclose true at1e−12. It is exact effective-kernel construction, not SVD compression.

context-blocks.py also executed. Its independent compact inverted-residual and ASPP-style definitions process real sourceIDs1/11 through randomly initialized blocks. Mobile output(2,8,8,8), context(2,4,8,8); a squared-output differentiation probe gives finite nonzero stem gradients. block-check-results.json stores actual shapes/flags. This is a building-block/gradient example, not fitted segmentation, benchmark weights or a prediction-quality experiment.

## Retention and phase-two work

The programs and results above are content-author calculations. Root content reconciliation, formal independent implementation review, browser rendering/accessibility, lazy-load/error behavior and integration are distinct. All runtime/production work is deferred. Do not rerun six fits for presentation-only edits; refresh relevant results after a substantive model/data/program change. Retain all listed files and license notices. No disposable own scratch or downloaded source media were created by this packet.
