# ConvNeXt data and calculation provenance

Research/write packet,13 September2026. All data/results/programs here are necessary pending implementation inputs; retain them. No browser implementation, pretrained download, GPU benchmark or full ImageNet reproduction was performed.

## Licensed real input and split

**Optical Recognition of Handwritten Digits**, E.Alpaydin and C.Kaynak(1998), [UCI record](https://archive.ics.uci.edu/dataset/80/optical+recognition+of+handwritten+digits), [DOI10.24432/C50P49](https://doi.org/10.24432/C50P49), distributed under CC BY4.0 in the UCI record inspected during this authoring run. Preserve attribution, license and transformation notice. No externally produced illustrative images were downloaded.

digits-400.csv is an unchanged copy of the attributed local subset used in the preceding Landmark and Depthwise packets. Select the first40 occurrences of each class0–9 from sklearn's1,797-row load_digits source and concatenate by class. source_id is one-based original row; digit is label; pixel_0…pixel_63 are row-major8×8 intensity0–16. This sklearn source derives from the UCI optical-digits test file; our teaching subdivision is not the original benchmark protocol and not MNIST. CSV SHA-256 a5b50ff0418e2b470140153a399c9200b2bba68468232507fd393ba89c4fc672,61,444bytes.

Fresh executed audit in masked-reconstruction.py before fitting:400 unique IDs and400 distinct complete pixel vectors. Writer/session identifiers are absent. This excludes exact image duplication but does not establish independent writers, related-image independence or deployment representativeness.

Fixed scaling pixel/16 uses the published intensity bound, with no fitted statistic. Stratified train_test_split(test_size=.3,random_state=22) gives280 training and120 development images,28/12 per class. Full source IDs and development labels are retained in calculated-inputs.json. No development images enter unlabeled pretraining or readout preprocessing/fitting. Development comparisons are consumed and exploratory; there is no untouched final test or benchmark-score claim.

## Executed learning program

masked-reconstruction.py ran with Python3.12.14, NumPy2.3.5, PyTorch2.14.0+cpu, sklearn1.9.1, one CPU thread and no dependency installs. Torchvision/timm were absent; PIL was available but unnecessary. Six fits: seeds1,2,3×GRN absent/present, each600 full-batch AdamW steps, learning rate.002 and weight decay.01 on all parameters, other optimizer defaults unchanged. Paired seeds and shared module order produce identical common initial tensors; mask RNG100+seed gives identical mask sequences. Additional GRN parameters start at zero without random draws.

Input shape N1×8×8. Binary visibility N1×4×4 selects6 of16 patches using rand/argsort; repeat each cell into2×2 pixels. Stem: Conv1→12 k2s2, channelLN12 epsilon1e−6, mask. Two encoder blocks: masked input, DW3p1 groups12 with bias, mask, channelLN, linear12→48 with bias, GELU, mask, optional dense per-image GRN48, linear48→12 with bias, residual addition, mask. A learned mask token fills inactive positions before one unmasked decoder block with the same small topology and no GRN, then a1×1 head12→4 and PixelShuffle2. Without GRN:4216 parameters; with GRN:4408. Both omit LayerScale, BN, DropPath and augmentation. Fits share neither trained weights nor pretrained state.

This small dense-masked single-stage model is not a ConvNeXtV1/V2/FCMAE reproduction. Differences:8×8 inputs,2×2 mask patches,62.5% hidden,3×3 DW, width12, two encoder blocks, raw scaled-pixel masked MSE,600 updates, decay on all parameters and a frozen linear probe. The paper uses32×32 input patches,60% masking, a full hierarchy, a sparse encoder or carefully masked dense alternative, patch-normalized targets and large-scale end-to-end fine-tuning. No large-scale result is inferred.

Masked MSE averages40 hidden pixels per image. For fixed comparisons, generate4 evaluation masks for all400 specimens with generator999 before fitting, then split the masks into train/development subsets. Evaluate at steps0,1,100,300,600 with eval/no_grad. Evaluation makes no updates; there is no early stopping or checkpoint selection. All-visible masks are used only for feature extraction, where masked MSE is not called.

Baseline: the training mean image, repeated over development examples and scored on the same4 masked target sets, MSE.07191353291273117. Raw-pixel StandardScaler+LogisticRegression(C1,max_iter2000), fitted only on training data: train280/280, development118/120, CE.11575614885911213.

After pretraining, clean all-visible encoder features flatten12×4×4=192. Freeze the encoder, fit StandardScaler and the same logistic regression using training labels. All six probe training counts are280/280. Final development results:

|Seed|GRN|MaskedMSE|Probe correct/120|ProbeCE|Mean channel cosine distance|
|---|---|---|---|---|---|
|1|absent|.05218800529837608|115|.12741687893867493|.4602116346359253|
|1|present|.052313175052404404|116|.12661418318748474|.462142676115036|
|2|absent|.05233616381883621|115|.10647080838680267|.49012789130210876|
|2|present|.05190592259168625|116|.10867030173540115|.4849971830844879|
|3|absent|.05167049169540405|115|.11663336306810379|.4854612350463867|
|3|present|.051488105207681656|116|.11455272138118744|.48374950885772705|

The feature diagnostic uses the clean-image final encoder expansion before GRN: channel spatial vectors within each image, norm threshold1e−8, separately reported near-zero fraction and mean(1−cos)/2 over valid nonself pairs. All near-zero fractions are0. This is neither the paper's visualized dead-channel count nor a supervised quality metric. GRN does not increase measured diversity in every seed. No random-encoder baseline was run, so these comparisons do not isolate pretraining's benefit.

Saved state: all metrics/predictions/traces, training mean image, two complete seed1 models, and the first two development examples source251/actual4 and source40/actual9 with original inputs, masks, raw reconstructions and individual MSE. Examples are the first two development rows, not selected for their errors. Display clipping must not change reconstruction loss.

## Independent author calculations, not formal phase-two review

author-checks.py imports the small model without running main/fitting. It reconstructs predictions independently using NumPy HWC scalar cross-correlation loops, explicit LN, scipy erf for exact GELU, GRN, masks, linear projections and PixelShuffle. All four defaults agree with PyTorchfloat32 and the stored JSON within3.20736e−7. This checks the learner model's arithmetic; browser implementation checks remain separate.

Executed interventions on every saved example: flip first hidden pixel(0,0),0→1, leaving all predictions exactly unchanged but changing target loss; flip first visible pixel(0,2) to1−old, changing prediction; swap the first row-major hidden patch with the first visible patch, preserving6 visible and changing prediction. Coordinates, masks and output changes are retained. Hidden-pixel invariance checks information availability, not accuracy. A display-label-only change is irrelevant because the model receives no label.

Constructed mathematical fixtures, explicitly not empirical observations:

- GRN channels A[3,4], B[0,12], gamma[.5,−.5], beta0; norms5/12, mean8.5. B12→0 demonstrates the shared denominator. Zero gamma/beta identity and zero-input null are checked.
- At gamma/beta0 with loss.5sumY²: gamma gradients[14.70588062284,203.2940937301], beta gradients[7,12], input gradient=input. Central-difference maximum error6.29207e−10.
- Axis fixture[1,3]/[101,103] compares channel LN with full-CHW normalization; three-channel[1,3,7]/[101,103,107] provides101→105 contrast and local+10 shift null(maximum change0).
- Compatible Conv3+Conv1+identity branches with fixed BN statistics fold to the stored kernel/bias; maximum error2.84217e−14. The branch-local ReLU counterexample differs by23.44987925 and explicitly negates the second branch's input.

The first author-check run completed calculations/assertions but JSON serialization failed on NumPy integer coordinates. Converted those to Python integers and saved again. A later bounded run added the checked three-channel contrast/null and patch swap; no fits were repeated. Final results come from the complete corrected script.

convnext-blocks.py is complete pedagogical V1/V2 block and hierarchy code with ChannelNorm, DropPath, GRN, initialization, feature outputs and a head. Ran meta shape/parameter checks for five V1 configurations and corresponding V2 constructions, plus actual2×96×3×3 block forward/backward for both versions with finite input gradients. V2 Small/XLarge here are generic constructed topologies, not claims about official named variants. V1 counts/MACs in the lesson agree with the helper. Block C96: V1=79,296,V2=79,968. Tiny's expected retained contributions are17.10000038 in float32, while all branches are calculated.

Meta full models do not allocate actual large weights or run large inference. No accuracy or timing is inferred from meta tensors. Conv/linear MACs exclude normalization, activation, bias, residual and memory costs; equal V1/V2 MAC counts do not mean equal runtime costs.

## Continuation

Preserve the CSV, provenance, programs and actual JSONs. Implement the visuals and investigations later using these model states and contracts; browser controls need no download or training. Formal independent review, validation of changed programs, browser numerical models, accessibility/mobile visual review, loading/error recovery and application integration remain not started. No GPU/mobile latency or rendered user approval is claimed.
