# PenDigits inputs and author calculations

This packet retains the original UCI Pen-Based Recognition of Handwritten Digits files by E. Alpaydin and F. Alimoglu. Dataset DOI: https://doi.org/10.24432/C5MG6K. Current repository: https://archive.ics.uci.edu/dataset/81/pen+based+recognition+of+handwritten+digits.

Retrieved 13 September 2026 from the official archive:
https://archive.ics.uci.edu/static/public/81/pen+based+recognition+of+handwritten+digits.zip.

The UCI page explicitly licenses the dataset CC BY 4.0: https://creativecommons.org/licenses/by/4.0/. Its original `pendigits.names` was read in full and contains no conflicting use restriction. Attribute the creators, link the source/license and identify transformations when publishing the interactive lesson. The author-created model programs and fitted arrays are separate from the byte-original source data.

| File | Bytes | SHA-256 |
|---|---:|---|
| Source zip, not retained | — | 1e02bea023613c2b11c9492f6f34caf975420455934f3527d270cee9a1f03b64 |
| pendigits.tra | 502,098 | e2b9eb9f0d0467e2b64a4816a3420edf2b8043447576f4b84337aba44a9f97d3 |
| pendigits.tes | 234,366 | 8bd03229c5c5291fefe43e45465dd948d2645bf23328b9d993e0b777666b2015 |
| pendigits.names | 4,953 | a90a76e1edac6f1b50f4b44db80587d7b92601fcbb153c83c2c3210c21b34f60 |

`source-download.json` records the retrieval URL and hashes. The original fields are sixteen integer coordinates in 0–100, interleaved x1,y1,…,x8,y8, followed by the digit 0–9. Coordinates were normalized and spatially resampled by the dataset creators. They are not an eight-frame image or eight evenly timed observations; pressure and original timestamps are absent. Preserve point order and do not invent pen-lift locations. The eight-point path is only the released approximation to the original handwriting.

## Frozen protocol

The official development file has 7,494 examples from 30 writers; the official assessment file has 3,498 from 14 other writers. This writer separation is described by the donor, while per-row writer identities are unavailable. Our classwise development split is not independently writer-grouped.

The author program checks exact coordinate tuples jointly across both source files before splitting. There are 10,992 unique coordinate sequences, no repeated feature groups, no label conflicts and no identical coordinates crossing the official boundary. Its general exclusion branch would exclude cross-file or label-conflicting groups and retain the first within-file duplicate, but it removes **zero** rows in this dataset.

Using NumPy default_rng(181), shuffle eligible development row IDs separately for digits 0 through 9. For each digit use the first 100 for fit and the next 30 for validation. Thus fit=1,000 and validation=300, both balanced. The other 6,194 development examples stay unused. All 3,498 official assessment rows remain for assessment. Exact 1-based IDs appear in `stroke-results.json:data.ids_1_based`; every figure and investigation source ID refers to the original file, not a reordered index.

Assessment class counts for digits 0–9 are [363,364,364,336,364,335,336,364,336,336]. The source normalization plus our fixed x/50−1 transformation does not learn from held-out rows. The neural position features t/7 and (t/7)^2 use the fixed complete eight-point grid even for prefixes and resumed chunks.

The six fits were declared before inspecting outcomes: linear-37, MMM-37, AAA-37, MAM-37, AMM-37, MAM-73. Neural width16, state4, three causal layers, convolution3 with previous-two-input buffers; one-head attention; dense SwiGLU width32; no MoE training, dropout, schedule search, augmentation or pretrained initialization. A separate linear baseline sees all sixteen flattened coordinates. Adam(.003), batch100, clip1, 80 epochs; each model gets the same batch order from a separate torch Generator seed491. Initial parameter seed differs as named. Lowest validation cross-entropy selects an epoch, with the earliest retained on a tie. Assessment was evaluated after checkpoint selection and all results are retained. There was no refit to improve the assessment ranking.

The primary development role is teaching causal information flow and exact cache continuation. This is a small real-data diagnostic, not a language-model reproduction or estimate of 256K-context quality/latency. MAM versus AMM has the same parameter count but changes mixer arrangement. MMM and AAA have different counts and operators. Two MAM seeds illustrate variation; they do not establish a seed distribution or statistical superiority.

## Reproduction and evidence files

- `stroke_models.py`: complete dataset parsing, duplicate check, deterministic split, all candidate models, full and incremental paths, fit/selection/assessment and saved-state loader. Default mode prints the existing report; `--train` explicitly reruns all six fits.
- `stroke-fits.npz`: selected parameter arrays for all candidates. Numeric arrays only; loaded without pickle.
- `stroke-results.json`: environment, exact IDs, histories, parameter counts, confusion matrices, full validation logits and all neural validation-prefix logits. Author evidence; do not automatically bundle this whole file into the page.
- `inspect_stroke.py`: complete executable inspection of one source row, optional deliberate cache/offset faults and all output probabilities. No network or retraining.
- `hybrid_mechanisms.py` and `mechanism-results.json`: original scalar read/collision, byte arithmetic, attention operation counts and Jamba-style/non-Jamba-style router comparisons.
- `author_calculations.py` and `author-results.json`: reload parity for every validation row, full/incremental differentiation, future-edit causality and checked worked/fresh/cache/null/input-edit fixtures.
- `packet_checks.py` and `packet-checks.json`: final source-baseline, syntax, lesson-link/disclosure and exact boundary checks, including recurrence/router identities and matching-carry cache nulls. It does not fit models or execute the large-model deployment example.
- `deployment_example.py`: separately labelled, unexecuted large-model example. It is not a dependency of the small experiment.

Actual environment: Python3.12.14, NumPy2.3.5, PyTorch2.14.0+cpu, one torch CPU thread, deterministic algorithms enabled. These computations ran in the existing read-only shared lesson runtime. No dependencies were installed into it. Float32 trained-model discrepancies have normal reduction-order effects; the independent float64 full/stream comparison is recorded separately.

Worked trace is the first selected validation example (class0, source2452). Fresh investigation is the second selected validation example of class1 (source2970). The initial first-two-row choice yielded two class0 examples; the fresh selection was changed to cover a second class, before writing its expected feedback. No fitting changed. Both cases retain faults that leave the final label unchanged as well as the fresh case where clearing K/V changes it. The edited-point contrast replaces point4 by (50,50); it is a learner perturbation, not another labelled observation. All-(50,50) is an intentionally degenerate input and has no ground-truth digit.

## Phase-two retention and delivery

Retain these inputs, arrays, specifications, manuscript and evidence until implementation accepts them. Later publication should lazily load only the selected small model/input fixtures for an opened investigation, with a bounded eight-point inference worker or equivalent off-main-thread calculation. Keep full research data and all training evidence as optional downloads rather than mandatory page payloads. Do not download large foundation-model weights or implement the model in phase one.
