# Optical digits: data provenance and experimental roles

Prepared 13 September 2026 for Modern Hopfield Networks, content phase only.

## Original source and permission

Dataset: Optical Recognition of Handwritten Digits, UCI dataset 80, E. Alpaydin and C. Kaynak, Department of Computer Engineering, Bogazici University, July 1998. Original work includes Kaynak's 1995 master's thesis and Alpaydin/Kaynak's Cascading Classifiers research.

- Dataset and current license: https://archive.ics.uci.edu/dataset/80/optical+recognition+of+handwritten+digits
- Official ZIP: https://archive.ics.uci.edu/static/public/80/optical+recognition+of+handwritten+digits.zip
- License: Creative Commons Attribution 4.0 International, https://creativecommons.org/licenses/by/4.0/
- ZIP SHA256: 0d7b054fea010270e9b3f06411c654c5e59547732ad626381980baffe0a23fb0

The UCI page was read on the preparation date and explicitly permits sharing/adaptation with attribution. The entire original optdigits.names file was read before retaining data; it contains no conflicting use restriction. We retain byte-original optdigits.tra, optdigits.tes and optdigits.names, not a re-labeled extract. The ZIP and temporary source download are not required retained assets.

| File | SHA256 | Role |
| --- | --- | --- |
| optdigits.tra | e1b683cc211604fe8fd8c4417e6a69f31380e0c61d4af22e93cc21e9257ffedd | 3,823 original training images |
| optdigits.tes | 6ebb3d2fee246a4e99363262ddf8a00a3c41bee6014c373ed9d9216ba7f651b8 | 1,797 original test images |
| optdigits.names | 3e82f7202d72a2b7dbdbc324c8c90fe8853164f5d6ab978a071357ad3de89f02 | Original description, attribution and acquisition |

Original data are unchanged. The lesson's derived role assignments, normalized values, occlusions, learned projections and measured results are modifications produced for this lesson; they are not the original authors' benchmark. Credit the original collectors and UCI beside downloadable data and the digit investigation.

## What one row means

Each row contains 64 integer input features in [0,16], followed by a digit label 0–9. The original normalized 32×32 bitmap is divided into 4×4 nonoverlapping blocks; each 8×8 cell counts on-pixels in one block. Display features in row-major 8×8 order. Intensities are not RGB colors or calibrated physical measurements.

Thirty writers supplied the original training split; thirteen different writers supplied the original test split. Per-writer identifiers are absent from these 65-column files. Do not claim person grouping within our fitting/validation split. The original named file describes a different historical within-training protocol; our deterministic role protocol below replaces that experiment, while preserving the separate-writer test boundary.

## Author protocol

digit_memory.py reads full source files. An exact-feature audit finds 3,823 distinct training vectors, 1,797 distinct test vectors, and no cross-split feature duplicates. All 5,620 rows have distinct 64-feature tuples. No rows are removed by the duplicate guard in this dataset. The program retains that check so source changes cannot silently insert exact self-matches.

For each class in order 0–9, permute its original training indices using one NumPy default_rng(113) stream. First 20 become fixed memories, next 80 fitting queries, next 30 validation. Roles contain 200/800/300 rows; the remaining 2,523 training rows are unused. All 1,797 test rows are assessed. Source IDs are one-based line numbers in their own original file; IDs from train and test are not interchangeable.

digit-results.json records every role ID. Only the 200 memory labels are read by inference. The 800 fitting labels optimize a shared projection; validation labels select hyperparameters/epochs; test labels contribute only to reported assessment. Query/reference pixels are divided by 16 and then mapped or normalized as described in the manuscript. Clean pixels remain available only for reconstruction evaluation.

Fixed geometry uses cosine-similarity keys. Select β from 4,16,64,256 by clean validation cross-entropy (selected 64). Nearest-memory cosine is a separate baseline. The script computes each predefined candidate's test report as evidence, but the selection expression uses validation alone; no candidate grid or model was altered after looking at test results.

Learned geometry uses a bias-free 64→16 shared projection with unit normalization, fixed β=16, cross-entropy over summed memory-label mass, Adam learning rate .005 and 100 full-batch updates. Seeds 17/41 select epochs 100/25 by clean validation cross-entropy. Seed 17 was predefined as the illustration run. No pretrained weights, pixel-loss training or masked training are used.

The fixed stress test zeros columns 4/5 of the 8×8 query grid. This makes 16 cells background without supplying a missingness indicator. Training and model selection use clean inputs. Reconstruction uses the same association weights on original memory pixels; it is not the optimized objective.

## Retained evidence and future packaging

- digit_memory.py: complete offline training/evaluation program.
- digit-results.json: source IDs, hashes, candidate results, training curves, confusion matrices and reconstruction metrics.
- digit-memory-fits.npz: both selected projections, the 200 memory images/labels, 300 validation images/labels, their class log-probabilities and association weights, and fixed-geometry validation results.
- author_calculations.py / investigation-checks.json: exact edited-image fixtures, restore and blank nulls, source IDs and full weighted images.
- associative_memory.py / mechanism-results.json: separate constructed mathematical examples. They are not real digit measurements.

Both selected CPU fits were executed. No browser models, pretrained benchmark reproductions, GPU performance comparison or independent implementation review was performed. Phase two may derive compact topic-owned lazy assets from retained data, with attribution and hashes; it must not bundle all raw source and author metadata into initial page load. Preserve this complete pending packet until implementation is finished and its retention decision is explicit.

