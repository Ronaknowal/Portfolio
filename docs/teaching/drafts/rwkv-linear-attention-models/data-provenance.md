# Libras Movement — source, roles and retained calculations

Research/reuse date:13 September 2026. Source [UCI Libras Movement, dataset181](https://archive.ics.uci.edu/dataset/181/libras%2Bmovement), DOI[10.24432/C5GC82](https://doi.org/10.24432/C5GC82). Credit Daniel Dias, Sarajane Peres and Helton Bíscaro, University of São Paulo,2009. UCI declares [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Preserve attribution, license link and transformation notice in all learner downloads and runtime exports.

The author read the current UCI page and original complete names file during this same-day sequence task. The self-attention author acquired the official [archive](https://archive.ics.uci.edu/static/public/181/libras+movement.zip) and permitted byte-original reuse. This packet copies its source files unchanged through the already retained SSM packet; it does not claim a second network download. The original names file has no conflicting use restriction.

| Source object | SHA256 |
| --- | --- |
| Official archive, acquisition author's digest; not retained redundantly here | 486f22f63919002a991be4042d7c7489a56cbd5f40f2930a24f80744c93e7105 |
| movement_libras.data | 97ebdaa6a9b28ab4a2cdd84b14f19a95a7456a46137c362b65a0669eca3c3c4d |
| movement_libras.names | 1b0702f0b664c84e66a908f07a967e56f18aea2b8a85d40f44a0d1cc2fd18c35 |

## Data meaning and preparation

360 rows,24 in each of 15 classes. Each row contains 90 ordered features x1,y1,...,x45,y45 and a label1–15. Metadata describes selecting45 frames from time-normalized videos of about 7 seconds and representing the hand centroid in unit space. Four performers and two sessions contributed; row-level performer/session IDs are absent. Do not claim held-out-performer/session transfer, calibrated physical speed or full sign-language translation.

Labels from the source:1curved swing;2horizontal swing;3vertical swing;4anti-clockwise arc;5clockwise arc;6circle;7horizontal straight-line;8vertical straight-line;9horizontal zigzag;10vertical zigzag;11horizontal wavy;12vertical wavy;13face-up curve;14face-down curve;15tremble. Python labels subtract1.

Preparation groups exact float64 coordinate tuples, verifies labels agree within each group, and retains the first occurrence. There are 330 unique trajectories and 30 duplicate copies. Using default_rng(73), permute retained IDs within each class; first floor(2n/3) fit, last 4 assess, middle validate. Totals220/50/60; roles disjoint and exhaustive over 330 retained rows. The raw files retain all 360 for provenance. Fixed2x−1 normalization converts tofloat32, with no fitted preprocessing or ordinal tags. Exact roles and duplicate groups are in trajectory-results.json.

## Actual executed experiment

trajectory_memory_models.py implements two small classifiers using the same data and declared protocol. Both project2→16, apply two residual blocks, average45 feature vectors and map16→15 logits. RWKV-4-style uses stable per-channel weighted state; positive-kernel uses ELU+1 features and normalized matrix state. Both use separate time/channel token-shift slots, sigmoid-parameterized mixing and a gated squared-ReLU channel branch with expansion 2. These are explicitly small teaching architectures, not research-checkpoint replications.

Adam learning rate .003,100 full-batch epochs, seeds 17/41, best validation cross-entropy epoch per run. No hyperparameter or seed selected after assessment. Ordered-coordinate C=1 logistic baseline has 1,365 coefficients/intercepts. RWKV-style5,263 parameters; kernel5,199. Errors fit/validation/assessment:
- Baseline35/17/22.
- RWKVseed17:31/18/19,epoch 67; seed 41:46/23/23,epoch 61.
- Kernelseed17:70/31/34,epoch 54; seed 41:34/22/23,epoch 69.

Environment Python 3.12.14,NumPy 2.3.5,PyTorch 2.14.0+cpu,scikit-learn 1.9.1; two CPU threads, deterministic algorithms. The four small fits were executed. No large pretrained checkpoint, GPU kernel, FLA implementation or browser model was executed. The real-data comparisons are limited row-level educational measurements; neither equal model capacity nor deployment generalization is established.

trajectory-results.json stores exact roles, duplicates, histories, selected epochs, parameter counts, predictions/confusion matrices and chunk-continuation differences. trajectory-memory-fits.npz stores selected numerical parameters/logits plus baseline coefficients; load with allow_pickle=False. Logits include all 360 original rows for traceability, but role metrics use only 330 deduplicated examples. Browser investigation needs only validation rows and selected frozen arrays.

author_calculations.py and investigation-checks.json store predetermined source 7 worked contrasts and fresh source 20 investigation defaults. They include point reflection, reverse-order, state-carry/reset, future-prefix causality, restored-input null and an autograd delta-gradient calculation. A small author helper variable-shadowing error was corrected before freeze: fresh-row editing originally overwrote the name used to report row 7's coordinate; final evidence correctly records+.2495200038→−.2495200038. It did not affect training or model results.

linear_memory_mechanisms.py and mechanism-results.json are original bounded mathematical examples, not sampled dataset observations: normalized-kernel direct/chunk equivalence, different-kernel contrast, stable RWKV reads/writes, additive/delta updates, RWKV7 core matrices, gated-memory outputs and calculated cache bytes. Label these exact calculations separately from fitted-data outcomes.

## Retention and phase two

Retain these source files, complete instructional programs, selected weights, results and manuscripts as necessary pending content artifacts. Delete only disposable own caches. Phase two will convert required arrays into lazy topic-owned runtime data, implement the specified figures/labs, independently verify the port and review browser/accessibility/performance/learning behavior. This packet does not publish or implement a website lesson.

