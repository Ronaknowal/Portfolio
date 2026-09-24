# Data, programs and calculation provenance

Research/write revision, 13 September 2026. Retain all files in this packet: they are pending implementation inputs, not disposable scratch. No model is trained or downloaded by the website in this revision.

## Real input and rights

E. Alpaydin and C. Kaynak (1998), **Optical Recognition of Handwritten Digits**, UCI Machine Learning Repository, [DOI 10.24432/C50P49](https://doi.org/10.24432/C50P49), [dataset record](https://archive.ics.uci.edu/dataset/80/optical+recognition+of+handwritten+digits). The UCI record was inspected during this topic's research on 13 September 2026 and explicitly identifies CC BY 4.0, the collectors and the original writer-pool boundary. Preserve attribution, license link and transformation notice. The retained original `optdigits.names` has no conflicting restriction.

The unchanged files were copied from the verified, already retained `xlstm-extended-lstm` packet rather than downloading duplicate archives. This topic independently inspected the UCI page, the full `.names` text, byte hashes and all vectors. The original collection normalizes 32×32 bitmaps, then counts marked pixels in each nonoverlapping 4×4 block to form an 8×8 array of integers 0–16. Inputs are fixed counts, not pixel coordinates, time samples, MNIST images or natural RGB photos.

| File | Bytes | SHA-256 |
| --- | ---: | --- |
|optdigits.tra|563,639|e1b683cc211604fe8fd8c4417e6a69f31380e0c61d4af22e93cc21e9257ffedd|
|optdigits.tes|264,712|6ebb3d2fee246a4e99363262ddf8a00a3c41bee6014c373ed9d9216ba7f651b8|
|optdigits.names|2,439|3e82f7202d72a2b7dbdbc324c8c90fe8853164f5d6ab978a071357ad3de89f02|

Actual audit: 3,823 distinct complete training pixel vectors; 1,797 distinct test vectors; zero exact matches across pools, no conflicting duplicate labels. The collectors state 30 training writers and 13 different test writers. Individual writer IDs are absent, so related specimens cannot be grouped inside the training pool. No additional segmentation, correspondence, medical or photographic labels were invented.

The supplied data is familiar from other educational packets. “Test” means held out from this topic's fitting/selection under its declared protocol; it is not a claim that the source has never been examined elsewhere in the curriculum. Preserve this distinction if repurposing the results.

## Declared study and executed fits

`experiment-contract.md` was written before fitting. `vision-study.py` contains the complete executable method; importing its definitions does not fit. NumPy default_rng(173) permutes each digit's original training-file rows; first 120 per class are training and next 30 validation. All other original training rows are unused. Store one-based source IDs in `study-results.json.split_ids`; the original test file remains in original order. Source pool is part of an identity, so `tra:1` and `tes:1` are different specimens. Scaling is exactly count/16 and uses no fitted statistic.

Runtime: Python 3.12.14, NumPy 2.3.5, PyTorch 2.14.0+cpu, SciPy 1.18.1, scikit-learn 1.9.1. One CPU thread, deterministic algorithms for the training run, no package installation or shared-runtime change. SciPy is used only for the independent NumPy GELU calculation in author checks. No GPU timing, mixed precision or foundation-checkpoint inference was performed.

Three declared neural fits, each 600 AdamW updates with learning rate .002, weight decay .01 on all parameters, other optimizer defaults unchanged; batch 128 sampled with replacement using a fresh generator 301 for each model. CNN seed 211; plain ViT seed 223. No augmentation, learning-rate schedule, dropout, DropPath or pretrained weights. Validation CE at step 0 and every 50 updates selects the minimum, ties earliest. Test predictions are computed after selection and never feed that rule.

CNN: Conv1→12 k3p1+bias, ReLU, MaxPool 2, Conv12→24 k3p1+bias, ReLU, flatten 384→32+bias, ReLU, linear 32→10+bias. Total 15,386 parameters.

Plain ViT: Conv1→32 k2s2+bias, sixteen patch positions, CLS and its separate position; two blocks, each LN 32(epsilon 1e−5), combined QKV 32→96 with bias, four heads of dimension 8, row-softmax(QKᵀ/√8), output 32→32 with bias, residual; LN 32, FFN 32→64→32 with biases and exact GELU, residual. Final LN 32 and CLS head 32→10. Total 18,218 parameters. Separate CLS and CLS-position parameters are intentionally redundant in their forward sum but independently stored; parameter accounting includes both. Learned positions initialize Normal(0,.02); CLS/CLS position start zero; all ordinary layers use PyTorch defaults.

Distilled student: same common tensors copied from the plain model's *initial* state; new DIST and its position start at zero, and new DIST head starts as a copy of the original CLS head. Total 18,612 parameters. The teacher selected checkpoint is frozen in eval mode; its hard training labels disagree with 8/1200 ground-truth labels. Training loss is half true-label CLS CE plus half teacher-hard-label DIST CE; no label smoothing. Evaluation averages logits, matching the inspected official DeiT code convention. The original paper instead describes softmax-output fusion; the lesson preserves that difference. Extra token/head/loss interactions mean this is not a parameter-equal or teacher-information-only comparison.

|Model|Selected step|Train correct/1200|Validation correct/300|Test correct/1797|TestCE|
|---|---:|---:|---:|---:|---:|
|CNN|550|1192|293|1723|.1358698010|
|ViT|350|1192|285|1630|.3815183043|
|Distilled|600|1195|283|1658|.3075848818|

Four additional declared readout fits: StandardScaler+LogisticRegression(C=1, max_iter=2000), trained only on training features/labels. Inputs are raw 64 pixels or selected 32-wide model features, using CLS alone for both Transformer readouts. Test correct/CE: raw 1695/.1683116555; CNN 1725/.1264231205; ViT 1637/.4182319045; distilled 1648/.3494727314. The latter is not the original fused head. These encoders were supervised; no DINO pretraining or unsupervised improvement is claimed.

All three full selected states are retained in `vision-models.json` as numeric lists. `study-results.json` stores split identities, metrics, traces and predictions. The original neural fits and four readout fits ran once. Later checks reload the saved states; they do not refit. The browser should receive a derived small plain-ViT asset only when I2 needs it, not all original research files.

## Real visual inputs and author arithmetic

`author-checks.py` reloads all selected states and reproduces every recorded test argmax. For the first three test images it independently calculates the plain ViT using NumPy patches/cross-correlation, explicit LN, exact GELU via SciPy erf, QKV, softmax, residuals and heads. Maximum discrepancy: logits 3.920604e−6, attention 2.994439e−7 versus PyTorch float32. This is author arithmetic evidence, not independent phase-two review.

Examples are the first three test rows: labels 0,1,2. They were not selected by success or heatmap appearance. Their images, logits, probabilities, last-layer/head 0 CLS attention and final patch features are retained in `author-results.json`.

Interventions on fixed state:

- Content-only swap of patch 5/6 changes logits; moving patch and position together preserves CLS to rounding. Maximum joint errors for sources 1/2/3 are 8.34465e−7,1.43051e−6,1.19209e−6. Content-only maximum changes are 7.83983,5.62649,6.93287.
- Reflect pixel(2,4) by 1−x. Sources 1/2 have real changed predictions; source 3 has x=.5 and therefore no changed input/output. The first assertion assumed all three inputs would change; the explicit source 3 equality was discovered and retained as an unchanged-input null. The corrected author check separates that null from actual perturbations.
- Fresh graded source 3 changes a *different* pixel(2,3),.8125→.1875. Probability of original annotated class 2 changes .0189909637→.9967051148. This is a real saved-model intervention, not a new labeled image or evidence that the model's edited confidence is calibrated.
- PCA uses all 19,200 training patch vectors, training mean and covariance eigenvectors. Sign convention makes each retained axis's largest-absolute loading positive. Ratios .2271616250,.1676369857,.1049629935 sum .4997616042. The manuscript initially had a one-percentage-point summation typo; final wording is 49.98%. The common training basis and fixed display ranges apply to all images. No separate per-test-image fit.
- Cosine matching source 1 patch 5 against source 2 patch features ranks 10,2,14 with values .3620218008,.2958233101,.2895212311. This is a numerical nearest-feature observation, not verified semantic correspondence. Zero directions must be rejected if user edits produce them.

## Constructed mechanisms and fresh tasks

`vision-mechanisms.py` calculates constructed fixtures, not simulated training benchmarks. `mechanism-results.json` records outputs:

- Row-major 4×4 values 0–15, patch 2, fixed two-feature projection; unfold+linear exactly equals Conv2d with tied weights/bias.
- Shifted averaging reach computed by explicit region matrices and their product. At 4×4 with window 2 and shift 1, query (1,1) has 16 original sources; on 6×6, query (2,2) has 16 and corner(0,0) has 4. Wraparound diagnostic: source(5,5)=16 gives unmasked corner 4 versus masked 0.
- Complete readable rolled `ShiftedWindowAttention` includes QKV/output weights, actual per-head relative bias, conceptual boundary regions, padded-key mask, finite treatment of discarded padded queries, reverse roll and cropping. Independent full-grid query loops on seeded 5×7×4 input agree to 5.55112e−17. Input gradients are finite; nonzero bias gradient maximum .3156425884 verifies the table affects the score. The example also includes a complete V1-style block and patch merge, without claiming to reproduce a full named Swin checkpoint.
- Bicubic rectangular position resizing takes explicit old 2×3 and new 3×4 grids with two prefixes, `align_corners=False`; prefix values exact, unchanged-grid identity exact. The grid is interpolated in float32, then converted back, explicitly matching the helper.
- DINO worked probabilities/loss/gradient independently checked: loss 1.0489189856; analytic gradient[-.6388526872,.3553009001,.2835517871], central-difference maximum error 1.30675e−10. Two-view example excludes same-view pairs and has no teacher gradient. All-zero three-prototype collapse has loss log 3 and zero student gradient.
- MAC totals use one multiply–accumulate, FFN ratio 4, explicit dimensions and projection/pair work separation. They exclude non-matmul work and imply no milliseconds. The stem has 742,656 parameters and includes projection bias, CLS and all positions. Constructed fusion counterexample chooses different argmax under mean logits versus mean probabilities; finite grid search only located a clean explanatory case, not an empirical score.
- Gram fixtures use normalized feature geometry, unnormalized summed squared Frobenius loss; common rotation preserves Gram, individual feature edit changes it.

Fresh gated inputs differ from the worked examples and separate exercises. They were checked with fixed mathematical inputs or saved weights only:

- A changed patch `[2,1,4,0]→[2,1,2,0]` gives `[2.5,−2]→[2.5,0]`; the different patch `[1,1,4,1]` preserves the baseline output.
- The fresh Swin path uses source (0,0)=16 and destination (2,2): fixed-window output 0 versus shifted-window output 1. The distant source coefficient at (5,5) is 0.
- The fresh single-layer Swin bias case uses value 10 at (1,1). The output at (2,2) changes 2.5→4 when the specified offset bias changes by log 2; an edit at the distant (5,5) is a null.
- The fresh DINO gradient for student prototype 0 changes +.1997045→−.3819665 when teacher prototype 2's logit changes .4→0. A common +3 teacher-logit offset is a nontrivial null.
- The fresh four-feature Gram edit has summed squared difference 6; a common rotation preserves the Gram matrix and has loss 0.
- The fresh actual-image edit and its exact probability change are recorded above.

All answers stay hidden in the future UI until a committed prediction. The visual specifications identify each fixture, prediction and edit explicitly.

## Deferred optional program and retention

`inspect-pretrained-features.py` is a complete optional local-file program written against inspected Transformers DINOv2 documentation. It has not been executed with Transformers/Pillow/checkpoint weights here. It requires a user-prepared local official non-register `facebook/dinov2-small` checkpoint/processor and local images; it uses `local_files_only=True`, validates token/grid relationships and saves features/cosines. No fabricated expected prediction, downloaded model or remote photograph is retained. The small inline timm setup is likewise an unexecuted API example, explicitly marked in the lesson.

The final compact author checks are recorded in design.md: program/JSON parsing, local links, disclosure, printed hand code and exact source preservation. These checks do not execute the optional checkpoint program. Formal independent review, browser implementation/numerical equivalence, mobile/accessibility, optional-checkpoint execution, loading/payload and publication checks are phase two. No disposable files were created in shared scratch; only necessary topic-owned programs/data/results are retained.
