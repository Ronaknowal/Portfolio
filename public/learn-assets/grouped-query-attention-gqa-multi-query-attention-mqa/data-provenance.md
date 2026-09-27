# GQA/MQA — data, protocol and calculation provenance

Research/write phase, 13 September 2026. This packet contains required offline inputs and observed calculations for later implementation; no website publication, browser lab or GPU-serving benchmark was performed.

## Source data and redistribution

Libras Movement is credited to Daniel Baptista Dias, Sarajane Marques Peres and Helton Hideraldo Bíscaro. Cite Dias, D., Peres, S., & Bíscaro, H. (2009), UCI Machine Learning Repository, [DOI 10.24432/C5GC82](https://doi.org/10.24432/C5GC82). The [official UCI 181 page](https://archive.ics.uci.edu/dataset/181/libras%2Bmovement) specifies [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). The complete original `.names` metadata was inspected in the preceding source acquisition and has no conflicting reuse restriction. Reuse that source-bound evidence; this packet copies the original two files without a new download.

Original archive: [libras+movement.zip](https://archive.ics.uci.edu/static/public/181/libras+movement.zip), acquired 13 September 2026, 164,761 bytes, SHA-256 `486f22f63919002a991be4042d7c7489a56cbd5f40f2930a24f80744c93e7105`. The archive is not retained. Original file hashes:

| File | SHA-256 |
|---|---|
|`movement_libras.data`|`97ebdaa6a9b28ab4a2cdd84b14f19a95a7456a46137c362b65a0669eca3c3c4d`|
|`movement_libras.names`|`1b0702f0b664c84e66a908f07a967e56f18aea2b8a85d40f44a0d1cc2fd18c35`|

Both were copied byte-for-byte from the immediately preceding Positional Encodings packet. The `.names` file retains its original Latin-1 encoding and accented creator names. The dataset contains 360 rows, each with 90 alternating x/y features and a class label for one of 15 movements. The 45 coordinate pairs are sampled from recordings of roughly seven seconds; they are ordered, normalized hand-centroid observations, not exact supplied timestamps or complete signed-language sentences. Four performers and two recording sessions are described, but reliable per-row identities are absent.

## Data boundary and the changed teaching task

The complete program groups all 90-coordinate feature rows exactly and checks label agreement within every duplicate group. There are 330 unique trajectories and 30 additional copies. Retain each group's first occurrence. One NumPy generator with seed 73 permutes retained rows within each class in class order: first floor(two-thirds) training, last four test, remainder validation. Totals 220/50/60; exact one-based source rows and duplicate groups are saved in `author-results.json`. Whole trajectories and their duplicates cannot cross this boundary. A row-level test cannot establish new-performer/session performance.

Unlike preceding classification studies, this packet forecasts **next coordinates**. Inputs at positions 0–43 predict targets 1–44. Logical causal legality prevents reading a future target. Class labels only preserve the split; they are neither inputs nor targets. Fixed transformation `2*x-1` is applied to x/y inputs and targets. Model MSE averages both transformed coordinates and all 44 output positions; original-coordinate RMSE is `sqrt(MSE)/2`. No test statistics or learned test normalization are used. Reusing the same source/split supports continuity, not an architecture ranking across the different earlier tasks/protocols.

Persistence predicts the newest input unchanged. An affine two-coordinate map with bias is fitted by NumPy least squares on training transitions only, with six coefficients. The same validation/test slots are scored. Baseline test original-coordinate RMSE is 0.0264584944 for persistence and 0.0262221852 for affine.

## Declared neural comparison and actual results

The experiment declaration was saved in design before fitting. One seed-101 parent uses a 2-to-24 stem, one causal pre-normalized block, four query heads of width six, four initial KV heads, adjacent-pair full Q/K RoPE base 10000, biased maps, FFN 48 GELU, final LayerNorm epsilon 1e-5 and a two-coordinate output per row. No dropout or weight decay. Train 180 full-batch Adam updates at 0.003; choose lowest validation MSE, earliest exact tie. The chosen parent update is 171.

Create three branches from that one selected parent: unchanged MHA continuation, group-mean GQA with two KV heads and group-mean MQA with one. Average K/V weights and biases within contiguous groups; copy every other parameter. Each branch receives 45 additional full-batch Adam updates at 0.001 using a fresh optimizer. Select lowest validation MSE, including zero updates. Test metrics are reported after selection; zero-conversion metrics are also retained to show the boundary. No test-driven schedule extension or additional seed campaign followed the results. Equal update counts are not asserted to equal large-model pretraining FLOPs.

| Branch | Parameters | Selected extra step | Before test RMSE | After test RMSE | Selected validation RMSE |
|---|---:|---:|---:|---:|---:|
|MHA, Hkv4|5042|41|0.0157894269|0.0156170319|0.0152602248|
|GQA, Hkv2|4442|45|0.0771819507|0.0182678328|0.0184376288|
|MQA, Hkv1|4142|42|0.1151743951|0.0252240072|0.0250899434|

These are bounded actual forecasting outcomes, not published language-model reproduction or a proof of final converged quality. The GQA selected step is the allowed budget endpoint. All outcomes, including MHA's advantage and converted models' initial losses, remain in the lesson.

Recorded environment: Python 3.12.14, NumPy 2.3.5, PyTorch 2.14.0+cpu, one CPU thread and deterministic algorithms. Shared workspace runtime was used read-only; no dependencies were installed and no GPU or network access is needed by the reproduction program.

## Saved model states and intervention scope

`forecast-models.json` retains the three **selected continuation checkpoints**, their real weights, source-row-77 prefix and actual raw Q/K, rotated Q/K, compact values, weights, head outputs and forecasts. It does not export the parent or zero-update conversion weights as separate interactive models; their aggregate metrics and full reproduction procedure are retained. Do not fabricate a zero-update trace by using a selected model's weights.

Source row 77 was declared before fitting as the first held-out class-4 record, regardless of outcome. Its first 32 points are observed; point 32 is the next target. At this boundary actual next point is `[0.5938100219,0.25]`. Selected MHA/GQA/MQA predictions are respectively `[0.6009761095,0.2584558427]`, `[0.5974121094,0.2590175867]`, `[0.6071050167,0.2532032132]`.

Each K and V tensor has shape `[1,Hkv,32,6]`; actual float32 payload totals 6144/3072/1536 bytes, excluding IDs and other model state. Native attention uses a query-group reshape and `einsum`, not an explicit repeated KV cache. Full sequence versus incremental processing agrees within 2.7e-7 transformed-coordinate maximum difference; all-position +100 shift agrees within 2.4e-7. Reflect observed frame 23 x→1−x and recompute the prefix: actual changed forecasts are saved. An independent post-export model check confirms that predictions before frame 23 remain exactly unchanged for all three models and that reloaded weights reproduce saved next forecasts.

The five-point rollout starts from the original prefix and feeds each generated point back as the next input. No true future coordinate enters that calculation. True future points remain available for later display from the original offline source, not as rollout inputs. Coordinates are not silently clipped into 0–1. User-edited examples do not gain new ground-truth labels or become members of an accuracy table.

## Independent mechanisms and author checks

`mechanism-calculations.py` contains a full NumPy grouped-attention reference and exact constructed fixtures: four-reader/two-memory weighted sums; shared key/value/query edits; consistent group relabeling; wrong routing; tied-MHA null; nonlinear mean-conversion counterexample; cache-byte counts and changed-practice values. It also runs small CPU float64 PyTorch checks for direct-grouped versus SDPA `enable_gqa=True`, repeated storage allocation, summed per-read value gradients and non-square mask contrasts. Its output is `mechanism-fixtures.json`. None of these constructed arrays is represented as a trained activation or measured throughput.

The first native derivative check emitted a layout warning because NumPy singleton-axis zero strides were retained when creating leaves. Constructing the small tensors from their value lists corrected the leaf layout; the scoped rerun passed with identical mechanism outputs and no warning. No model fits were repeated for that fix.

The manuscript's full NumPy program was separately extracted and executed as displayed: output table and tied-MHA/compact-cache checks passed. Saved models were reloaded for bounded forecast/causality checks. Final author continuity, links, source hashes and disclosure checks are recorded in design; formal independent/rendered/browser review remains phase two.

## Retention

Retain all eleven files: manuscript, visual specifications, design, this provenance, original data and metadata, complete study program/results/model states, and independent mechanism program/results. They are required pending implementation inputs. No new disposable scratch was created. Phase two derives small per-model semantic assets and optional reproduction downloads; no eager full training-history/model-evidence import is needed. Preserve attribution, exact input boundaries and observed-versus-derived labels. Root owns shared ledger/hash closure.
