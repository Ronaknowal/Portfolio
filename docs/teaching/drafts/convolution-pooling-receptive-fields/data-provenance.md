# Convolution experiment provenance

Research/write packet prepared 2026-09-12. Retain all seven packet files for deferred implementation; no website runtime is changed.

## Real inputs

digits-400.csv contains 400 actual records from scikit-learn load_digits, whose 1,797-row copy comes from the historical UCI optical-digit test partition. Original creators: E. Alpaydin and C. Kaynak, 1998. Dataset: [Optical Recognition of Handwritten Digits](https://archive.ics.uci.edu/dataset/80/optical+recognition+of+handwritten+digits), [DOI 10.24432/C50P49](https://doi.org/10.24432/C50P49), CC BY 4.0. This is not MNIST.

The unchanged subset is the first 40 occurrences of each label 0–9, concatenated by label. source_id is the original one-based sklearn row; pixel_0 through pixel_63 are row-major 8×8 integer intensities from 0 to 16; digit is the target. CSV SHA-256 a5b50ff0418e2b470140153a399c9200b2bba68468232507fd393ba89c4fc672. Preserve attribution and disclose this subset transformation.

Scale by the known upper measurement bound 16, then stratified train/development split with random_state 22, sizes 280/120. Both original source-ID arrays are retained. No fitted transformation consumes development rows. The split is not an official UCI score, independent-writer evaluation or untouched final-test protocol. The shared examples help continuity with preceding neural packets, but are not evidence that a large benchmark is solved.

## Executed experiment and measured outputs

convolution-experiments.py ran with Python 3.12.14, torch 2.14.0+cpu, NumPy 2.3.5, scikit-learn 1.9.1, one torch CPU thread. Shared environment read-only; no installation or external pretrained download.

Twelve actual fits: seeds 1/2/3, four configurations each. Dense64→32 tanh→10, 2,410 parameters; CNN conv1→8 k3/p1/ReLU/pool2 then conv8→16 k3/p1/ReLU/pool2 then flatten64→10, 1,898 parameters, using max or average pooling; max CNN with final map mean and16→10 head, 1,418 parameters. Max and average variants start identically within seed. Global-average variant shares initial convolution tensors but has a different-sized head. Dense baseline has different weights/architecture; parameter counts are not matched.

Adam .003, 400 full-batch steps, no weight decay/dropout/augmentation/early stopping. Train/development mean cross-entropy and correct/count recorded in eval/no_grad at steps 0/1/25/100/200/400. Every final model correctly classifies all 280 training rows. Seed1 final CE/correct: dense .079494476/117, max .083323374/118, average .103576228/116, global average .082494326/117. All three seeds and traces remain in calculated-inputs.json; no interpolation is claimed as a measured step.

Predeclared stress views translate development images one pixel right/down, zero fill and crop to 8×8, retain labels. They can remove strokes and do not isolate ideal translation equivariance. All twelve aggregate scores are saved. No shifted-specimen logits or maps were retained. No stress data is used in fitting, but their inspected outcomes are development evidence for any later decision.

For the three seed1 CNNs, saved source IDs 251/40/149 (digits4/9/8) have original scaled input, ten probabilities, post-ReLU first maps8×8×8, post-ReLU second maps16×4×4, final pooled maps16×2×2, and signed input gradient of the currently predicted logit. All three are correctly predicted in these saved cases. First-layer weight grids are saved; first-layer biases, complete final parameters and raw preactivations are not. Do not synthesize maps for other seeds, arbitrary edited inputs, stress images or logits from kernel images alone. Re-running the complete program is the explicit route to new model outputs.

## Exact mechanisms, calculations and limits

Executed float64/NumPy/PyTorch arithmetic: sliding patch and unfolding; overlap-count reconstruction; five ordinary/strided/dilated/grouped/depthwise-multiplier parity cases; one shared-filter update and input gradients; four changed-target/rate/null updates; matrix-adjoint inner products and noninverse example; pooling overlap/ties/negative padding/average denominators/adaptive bins and gradients; circular/zero boundary contrast; alternating-signal aliasing/lowpass; full receptive-coordinate trace; exact finite linear profiles; dilation holes; transposed overlap counts; fixed inference BatchNorm folding; even/asymmetric shape/center cases; zero filter, constant derivative, closed ReLU and permuted mean nulls; changed center and diagonal-filter outputs.

NumPy-versus-torch maximum error across five cases 5.33e−15; inference BN folding error3.33e−16. These are bounded CPU discrepancies, not claims of universal exact floating-point agreement or hardware speed.

Linear profiles are exact repeated convolution coefficients for the declared positive three-tap linear operator; depth20 support41,1%-peak width21. They are neither trained empirical receptive fields nor invented Gaussian measurements. The learned signed gradients are a different saved input/model-specific quantity.

The channel construction outputs, 1D offset unions, parameter/MAC budgets, antialias filter constraints and fixed temperature stencil are small derived calculations. The diffusion connection computes only spatial stencil numerator−40 on a unit-spaced hypothetical grid; no PDE time integration, physical experiment or learned simulation was run. No CUDA/cuDNN benchmarks, compiled kernels, FFT/Winograd experiment, real decoder training, augmentation trial, population uncertainty estimates or semantic explanation validation were performed.

Initial twelve fits executed once. Later additions reran only fixtures(), preserving recorded fitted outputs. Program and source outputs are retained; no disposable scratch or bytecode cache was introduced for this packet.

## Deferred work

Website publication, code-native figures, interactive state/grading, translated calculation parity, download placement, canonical-link integration, accessibility/mobile/performance/browser/build checks and formal phase-two review remain not started. Preserve packet inputs until that work finishes and its retention policy applies.
