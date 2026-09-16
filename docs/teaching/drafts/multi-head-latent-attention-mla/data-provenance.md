# MLA — source data, calculations and retained evidence

Research/write phase, 13 September 2026. No publication, browser implementation or GPU performance benchmark was performed. This packet retains the complete offline inputs and actual author calculations needed to implement its specifications later.

## Original observations and rights

Libras Movement is credited to Daniel Baptista Dias, Sarajane Marques Peres and Helton Hideraldo Bíscaro: Dias, D., Peres, S., & Bíscaro, H. (2009), UCI Machine Learning Repository, [DOI 10.24432/C5GC82](https://doi.org/10.24432/C5GC82). The [official UCI 181 page](https://archive.ics.uci.edu/dataset/181/libras%2Bmovement) specifies [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). The complete original `.names` metadata was inspected during the preceding source acquisition and contains no conflicting redistribution restriction. This packet reuses that source-bound rights evidence and copies the two original files byte-for-byte from the preceding GQA packet.

Original archive: [libras+movement.zip](https://archive.ics.uci.edu/static/public/181/libras+movement.zip), acquired 13 September 2026, 164,761 bytes, SHA-256 `486f22f63919002a991be4042d7c7489a56cbd5f40f2930a24f80744c93e7105`. The archive itself is not retained. Retained original hashes:

| File | SHA-256 |
|---|---|
|`movement_libras.data`|`97ebdaa6a9b28ab4a2cdd84b14f19a95a7456a46137c362b65a0669eca3c3c4d`|
|`movement_libras.names`|`1b0702f0b664c84e66a908f07a967e56f18aea2b8a85d40f44a0d1cc2fd18c35`|

The metadata remains in original Latin-1 encoding. The source contains 360 rows of 90 alternating x/y features and a label for one of 15 movements. Each row describes 45 ordered normalized hand-centroid points derived from a roughly seven-second recording. Supplied features do not include exact timestamps, a raw camera pipeline, complete sign-language semantics or reliable per-row performer/session IDs. Four performers and two recording sessions are described in the metadata. Do not claim new-performer evaluation or online normalization validity from these processed row-level observations.

## Exact data boundary

The complete study groups all 90-coordinate feature rows exactly, checks duplicate labels agree, retains the first row in each group and records duplicate source rows. There are 330 distinct trajectories and 30 additional copies. A single NumPy generator with seed 73 permutes retained rows within labels 1–15, taking the first floor(two-thirds) for training, last four for test and intervening rows for validation. Totals are 220/50/60. One-based exact source rows and duplicate groups are saved in `author-results.json`; whole trajectories cannot cross the boundary.

Inputs at positions 0–43 predict targets 1–44. A causal mask prevents reading the target or later observations. Class labels are used only for stratification, not as inputs or targets. Fixed `2*x-1` transforms coordinates; MSE averages the two transformed coordinates and all 44 output slots, then original-coordinate RMSE is `sqrt(MSE)/2`. No test-derived normalization, selection or adaptation occurs. The same data/split was used by previous topics, so these observations do not form independent new evidence across lessons and differently shaped models must not be ranked as one controlled comparison.

Persistence predicts the newest point unchanged. An affine coordinate map with six parameters is fitted by NumPy least squares using training transitions only. Test original-coordinate RMSE is 0.0264584944 for persistence and 0.0262221852 for affine.

## Declared trained model

The design declaration preceded fitting. One seed-131 model uses a 2-to-24 stem, one causal pre-norm block, four heads with content width 4, value width 4, rotary width 2, normalized KV latent width 8 and normalized query latent width 12. Latent RMSNorm epsilon is 1e-6 with trainable scales; residual/final LayerNorm epsilon is 1e-5. Attention maps are bias-free; stem/FFN/forecast maps have biases. The FFN has width 48/GELU; the task head has two unconstrained coordinates. Ordinary adjacent-pair RoPE uses base 10000, no YaRN. There is no dropout or weight decay.

Train for 200 full-batch Adam updates at 0.003; choose lowest validation MSE, earliest exact tie. Selected update 200 is the budget endpoint; no convergence claim or subsequent retuning is made. Total parameters 4,118. Recorded full-model original-coordinate RMSE: train 0.0262952968, validation 0.0255762966, test 0.0249463655.

After selection, stack all content-key and value up-projection weights into a 32-by-8 joint output-by-latent matrix. Its right singular vectors define a predeclared rank-4 projection of the **already normalized eight-coordinate latent**. Cache four projected coordinates, and right-multiply each up-map by the corresponding 8-by-4 basis. No weights are retrained. The original eight-coordinate RMSNorm and query/rotary paths remain unchanged. This is not a separately trained rank-4 model or an MHA-to-MLA conversion experiment.

Actual rank-4 RMSE: train 0.0935409619, validation 0.0954812309, test 0.0919342231. The joint-map singular values are approximately 1.6292901, 1.5270361, 1.3194628, 1.2884343, 1.0238905, 0.9340577, 0.7241777, 0.6461220. Squared Frobenius reconstruction error 2.8627224 agrees with discarded squared singular sum 2.8627222 up to float32 rounding. Retain the substantial degradation; no alternate rank or new schedule was chosen to improve the displayed result.

The complete rank-8 singular basis is an orthogonal coordinate change, not truncation. Its validation outputs differ from the original basis by at most 3.5762787e-7 transformed-coordinate units. Source-row and model selection are independent of these subsequent inspection outcomes.

## Actual execution, gradients and example controls

Both expanded-head and absorbed-latent paths are fully implemented in `author-calculations.py`; both retain the original score divisor sqrt(6). A float64 two-trajectory/seven-position check compares the full model and all corresponding parameter gradients of a squared-output loss. Maximum output error is 3.8857806e-16; maximum parameter-gradient error is 7.1054274e-15. Every parameter has a compared gradient. This is actual derivative equivalence evidence, not an assertion inferred from successful training.

Source row 77 and its first 32 points were predeclared before fitting. The actual next point is `[0.5938100219,0.25]`. Full-model next forecast is `[0.5997370481,0.2636269927]`; rank-4 forecast is `[0.6137580872,0.2609272301]`. `forecast-model.json` retains the model weights, full/reduced singular bases and actual selected-input traces through latents, Q, rotary fields, separate score terms, weights, mixed latents, head outputs and predictions.

Full-model float32 expanded/absorbed maximum prefix output error is 1.7881393e-7; incremental-cache versus full-pass error is 3.5762787e-7. For rank four these are 1.7881393e-7 and 2.6822090e-7. Uniform logical-position +100 shift gives 1.7881393e-7 error for both. Reflect observed frame 23 x→1−x and recompute: forecasts become `[0.6005192399,0.2634072304]` and `[0.6144915819,0.2612116635]`. Outputs before frame 23 remain exactly unchanged for both cases.

Using the wrong default divisor sqrt(latent width + rotary width) changes the full model's prefix outputs by up to 9.3251467e-5; its next forecast becomes `[0.5997453928,0.2636736035]`. Under rank four, latent and content widths coincide, so that same nominally wrong-default choice happens to equal the intended divisor and yields an exact null. The specifications retain this distinction instead of forcing every control to show a difference.

Actual compact float32 fields are `[1,32,8]` plus `[1,32,2]`, totaling 1,280 bytes; the rank-4 fields total 768 bytes. Position metadata and all other model state are excluded. Cached state is tied to its model, original observation prefix, basis and positional convention. Changed or generated inputs do not receive new ground-truth labels or aggregate benchmark scores.

Recorded environment: Python 3.12.14, NumPy 2.3.5, PyTorch 2.14.0+cpu; one CPU thread, deterministic algorithms. Shared workspace runtime was used read-only. No install, external model, GPU run or new disposable scratch was required. The sole declared fit and its bounded checks completed successfully.

## Independent constructed evidence

`mechanism-calculations.py` is a complete NumPy implementation and fixture generator. Its hand example deliberately uses a quarter-turn-per-position frequency for simple arithmetic; it is not the real model's ordinary RoPE frequency. It checks expanded/absorbed equivalence, shared latent versus head-specific maps, invertible latent-basis preservation, value-only and latent edits, a global position-shift null, a query-only shift contrast, rotary/projection noncommutation, a nonlinear value-map counterexample, rank-one logits with full-rank softmax and input-sensitive SVD truncation.

Payload and operation counts are derived from labelled tensor dimensions. They are not measured GPU allocation, bandwidth, throughput or timing. Exact results are retained in `mechanism-fixtures.json`. Published model settings and current kernel-format examples are linked to their original sources in the manuscript/design; they are not merged into a fabricated local quality/latency benchmark.

## Retention and deferred work

`practice-calculations.py` additionally computes fresh gated investigation inputs, saving author-only answers in `practice-fixtures.json`. These distinguish the already-solved manuscript walkthroughs from independent learner predictions. The program loads the existing selected model, changes the visible prefix to 27 points and reflects frame 19's y coordinate; no fitting, selection, new architecture score or additional source acquisition occurs. Fresh matrices/vectors and cache dimensions support the other four investigations. Exact controls, full-model outputs and earlier-output causal nulls were executed and passed; these answers remain hidden in the learner interface until committed prediction/reveal.

Retain all thirteen topic files: manuscript, visual specifications, design, this provenance, original data/metadata, complete trained-study program/results/weights, independent mechanism program/results and fresh-practice program/results. They are required pending implementation inputs. Phase two derives compact selected-input/model assets and optional full reproduction downloads, then performs formal independent review, numerical/browser/accessibility and integration checks. The author continuity and bounded link/hash/disclosure checks are recorded in design; no rendered acceptance is claimed by this phase.
