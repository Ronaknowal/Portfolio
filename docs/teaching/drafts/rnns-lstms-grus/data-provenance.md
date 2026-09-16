# Pen trajectory data and author calculations

Prepared 13 September 2026 for `rnns-lstms-grus`, content-first only.

## Source, permission and transformation

E. Alpaydin and F. Alimoglu (1996), **Pen-Based Recognition of Handwritten Digits**, UCI Machine Learning Repository, DOI [10.24432/C5MG6K](https://doi.org/10.24432/C5MG6K), [current dataset record](https://archive.ics.uci.edu/dataset/81/pen+based+recognition+of+handwritten+digits). UCI explicitly licenses this record under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Attribute the dataset authors and UCI, link the license, and identify this lesson's selection and scaling when republishing the extract.

Downloaded the public UCI archive into memory with `prepare-pen-data.py`. Read the entire `pendigits.names` documentation and current record's schema and license. The names documentation did not introduce a conflicting reuse restriction. It reports 7,494 training and 3,498 test specimens, with original writer pools of 30 and 14. Per-row writer IDs are not supplied; do not claim an independently verified count of writers inside this lesson's subset.

The source preprocesses completed pen traces into eight ordered coordinate pairs in the range 0–100. The points are resampled by distance along the trace; they are not equally spaced timestamps. Original raw sampling intervals cannot be attached to the eight resampled points. Pressure is not retained. Whole-specimen normalization/resampling prevents interpreting an intermediate readout as a causally preprocessed live prefix.

The extract scans the source files in their original order, retaining the first 60 unique coordinate vectors per digit from `pendigits.tra` and first 30 per digit from `pendigits.tes`. No candidate duplicates were skipped. The full source audit found all 7,494 training vectors unique, all 3,498 test vectors unique and zero exact coordinate vectors shared across the two files. This checks exact coordinate duplicates, not unavailable person identities or approximate copies.

`pen-trajectories.csv` contains 900 rows, a declared partition, original filename plus one-based source row, 16 coordinate columns and the digit label. The source's former test subset is called **development** here because model comparisons and perturbation probes inspect it. No untouched final test remains inside the retained subset. Selection is class-balanced and deterministic, not a random benchmark sample.

## Byte provenance

| Retained or downloaded object | SHA-256 |
| --- | --- |
| Public UCI ZIP | `1e02bea023613c2b11c9492f6f34caf975420455934f3527d270cee9a1f03b64` |
| `pendigits.tra` inside ZIP | `e2b9eb9f0d0467e2b64a4816a3420edf2b8043447576f4b84337aba44a9f97d3` |
| `pendigits.tes` inside ZIP | `8bd03229c5c5291fefe43e45465dd948d2645bf23328b9d993e0b777666b2015` |
| `pen-trajectories.csv` | `e9f7d82554a7704edd3d63b07f62b575572675bfa6927e8fea3c52ec879be82f` |

`data-extraction.json` records source row counts, uniqueness and selection counts. The archive itself was not retained; the extraction program, exact URL, source hashes and selected data are enough to reproduce and inspect the teaching input.

## Model protocol

Fixed coordinate scaling is `coordinates / 50 - 1`. No statistics are fitted using development data. Baseline StandardScaler and logistic regression are fitted on the 600 training specimens only.

`pen-sequence-learning.py` executes two predeclared baselines and nine recurrent fits. Each RNN/LSTM/GRU has one unidirectional layer, input 2, hidden 32, ten-class linear output and zero initial state per specimen. The native initialization seed is 1, 2 or 3; recurrent biases are zeroed, except the LSTM input-side forget slice is set to 1 and hidden-side forget bias remains zero. Readout initialization uses the native linear default. Architectures do not have identical initial weights.

500 Adam updates, learning rate .005, batch size 64 with replacement, cross-entropy, global norm clipping at 1, paired specimen-index generator seed `100 + seed`. No augmentation, dropout, weight decay, early stopping or run selection. Every 0/1/100/300/500 checkpoint is descriptive. All three seeds and both fixed baselines are reported.

Evaluation reverses all eight points or swaps points 3 and 4 with **fixed weights and retained original digit labels**. These are synthetic transformations of real inputs, not natural independently collected test sets or refitted reversed models. Seed-1 model weights and two development examples preserve actual recurrent states and readout probabilities. Prefix probabilities were not trained or validated as live prefix predictions.

## Execution and retained evidence

Author CPU environment: Python 3.12.14, NumPy 2.3.5, PyTorch 2.14.0+cpu, scikit-learn 1.9.1. One Torch CPU thread. The shared runtime was used read-only, without installations.

`calculated-inputs.json` stores versions, exact training/development IDs, labels, both baselines, all nine runs and saved seed-1 weights. `mechanics-results.json` stores independent NumPy state/gate/probability traces, actual changed-coordinate inputs, scalar BPTT, fixed-gate decay, a full-state Jacobian, reset-placement counterexample, clipping arithmetic and native state/padding fixtures. Fitted-model probabilities are not hand-drawn.

Single-layer parameter totals are RNN 1,482; LSTM 4,938; GRU 3,786, including the common 330-parameter readout. Native and independent NumPy traces agree within 3e-6; separate float64 nonzero-state/nonzero-bias cases agree within 1e-12. Carry-versus-whole and detach-forward nulls are exact for the deterministic fixture. Padding and gradient checks do not require repeating the nine fits.

Proposed plots have explicit categories: empirical results from these fits, exact constructed arithmetic, or symbolic diagrams. None is a latency benchmark, universal architecture ranking, live-prefix performance result or guarantee of long-range memory.
