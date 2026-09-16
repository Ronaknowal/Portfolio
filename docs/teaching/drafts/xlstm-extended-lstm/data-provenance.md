# Optical digits for xLSTM: provenance and experiment

Prepared 13 September 2026, research/write phase only.

## Source, acquisition and permission

Optical Recognition of Handwritten Digits, UCI dataset 80, was contributed by E. Alpaydin and C. Kaynak, Department of Computer Engineering, Bogazici University, July 1998. Dataset page: https://archive.ics.uci.edu/dataset/80/optical+recognition+of+handwritten+digits . Official ZIP: https://archive.ics.uci.edu/static/public/80/optical+recognition+of+handwritten+digits.zip . The UCI page explicitly identifies Creative Commons Attribution 4.0 International: https://creativecommons.org/licenses/by/4.0/ .

The original dataset and license were read in this same authorized session during the preceding Modern Hopfield packet; this packet copied its verified byte-original files, not a fresh download. The complete `optdigits.names` was reread for this packet. It has acquisition details and no conflicting use restriction. ZIP SHA256: `0d7b054fea010270e9b3f06411c654c5e59547732ad626381980baffe0a23fb0`.

| Retained file | Rows / role | SHA256 |
| --- | --- | --- |
| optdigits.tra | 3,823 original training images | e1b683cc211604fe8fd8c4417e6a69f31380e0c61d4af22e93cc21e9257ffedd |
| optdigits.tes | 1,797 original test images | 6ebb3d2fee246a4e99363262ddf8a00a3c41bee6014c373ed9d9216ba7f651b8 |
| optdigits.names | Original metadata and attribution | 3e82f7202d72a2b7dbdbc324c8c90fe8853164f5d6ab978a071357ad3de89f02 |

Each row has 64 integer features in [0,16] and a label 0–9. The features count on-pixels in nonoverlapping 4×4 blocks of a normalized 32×32 bitmap, giving an 8×8 grid in row-major order. There are no missing input values. These counts are not timestamps, continuous pen trajectories or calibrated image intensities. Our choice to process eight horizontal strips as sequence steps is a derived modeling choice.

The original training and test files come from 30 and 13 different writers respectively. Individual writer IDs are absent within each file. Our internal fit/validation split cannot claim separation by writer. The source's historical within-training experiment is not our experiment. All 5,620 feature vectors are distinct; no exact feature duplicate crosses any role in the supplied data.

## Exact derived protocol

`row_sequence_models.py` uses one NumPy default_rng(157) stream and iterates class labels 0–9. Each class's training-file indices are permuted; first 100 go to fit, next 30 to validation. Totals are 1,000 fit and 300 validation, with 2,523 unused training-file rows. All 1,797 test-file rows are assessed. Source IDs are one-based line numbers within their respective original file, saved in `row-sequence-results.json`; a train ID and test ID with the same number are not the same observation.

Inputs are divided by 16, with no learned preprocessing statistic. Labels optimize cross-entropy on the fitting set, select the minimum-validation-cross-entropy epoch, and measure the assessment set, respectively. Inference takes only pixels and optional recurrent state.

The complete instructional architecture is in `DigitReader`: row projection 8→16, pre-RMSNorm, selected sequence cell, residual, post-RMSNorm, SwiGLU-style expansion 16→32→16, residual, classifier 16→10. Only final-row logits enter fitting loss. Ordinary LSTM has 4,138 parameters; one-head sLSTM 4,074; one-head mLSTM 2,828, with key width eight and value width 16. They are not parameter-matched or reproductions of a large published model.

Each of three cell choices has two predeclared seeds, 19 and 43. Adam learning rate .003, 150 full-batch updates, gradient norm clipped at one. Checkpoint selection uses clean validation cross-entropy after each update. All six selected runs are reported, including unfavorable matrix-model outcomes. No new configuration was chosen to rescue that result after test inspection.

The fixed stress condition reverses the eight rows before processing while preserving each row's pixel order and the image label. Models fit only clean natural-order images. It is not a horizontal mirror, random pixel permutation or augmented training run. Additional single-image illustrations zero the final three rows or all pixels. They demonstrate causal computation on validation examples, not a new population-level robustness benchmark.

The code saves fit loss before the optimizer update and validation loss after the update for each epoch. A chart must label that difference. Per-prefix logits are computed for inspection; these prefixes were not separately supervised. Probabilities are not assessed as calibrated confidence.

## Evidence and retained assets

- `row_sequence_models.py`: complete offline data loading, training, selection, evaluation and state-carry checks.
- `row-sequence-results.json`: source hashes/IDs, all six curves, selected epochs, parameter counts, clean/reverse confusion matrices, cross-entropy and state/prefix checks.
- `row-sequence-fits.npz`: each selected parameter dictionary plus the 300 validation images/labels and all eight prefix logits for clean/reverse conditions.
- `memory_mechanisms.py` / `mechanism-results.json`: constructed scalar/matrix/operator examples and mathematical checks, not real-data model measurements.
- `author_calculations.py` / `investigation-results.json`: exact new scalar/matrix questions, one gradient step, changed practice calculation, and actual validation image fixtures with edited/reversed/blank/carry/reset outputs and state traces.
- `check_author_packet.py` / `author-checks.json`: bounded checks of the completed packet, added during author closure.

All six fits ran on CPU in Python 3.12.14 with NumPy 2.3.5 and PyTorch 2.14.0+cpu. No pretrained download, GPU benchmark, official-kernel execution or browser implementation was performed. Hardware speed ratios and large-model quality scores are not inferred from these tiny fits.

Credit the collectors and UCI beside the data download and digit investigation. The byte-original inputs are unchanged; normalization, row scanning, split roles, learned fits and edited-image results are lesson-derived modifications. Phase two should derive compact, attributed, lazily loaded teaching assets and preserve parity to these retained results. Do not eagerly ship the raw dataset, all curves or author evidence to every page. Keep this pending packet intact until implementation has consumed it and an explicit retention decision is recorded.
