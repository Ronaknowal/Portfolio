# Sparse & Linear Attention — data and calculation provenance

Research/write phase, 13 September 2026. Preserve these offline inputs for later implementation. This is not a website publication, browser review or GPU benchmark.

## Source and rights

Credit Daniel Baptista Dias, Sarajane Marques Peres and Helton Hideraldo Bíscaro: Dias, D., Peres, S., & Bíscaro, H. (2009), *Libras Movement*, UCI Machine Learning Repository, [DOI 10.24432/C5GC82](https://doi.org/10.24432/C5GC82). The [official dataset page](https://archive.ics.uci.edu/dataset/181/libras%2Bmovement) specifies [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). The complete original names metadata was inspected during the preceding source acquisition and contains no conflicting reuse restriction. This packet reuses those verified bytes and source-bound rights evidence rather than downloading the same source again.

Original archive [libras+movement.zip](https://archive.ics.uci.edu/static/public/181/libras+movement.zip), acquired 13 September 2026, 164,761 bytes, SHA-256 `486f22f63919002a991be4042d7c7489a56cbd5f40f2930a24f80744c93e7105`. The disposable archive is not retained. Original file hashes:

| File | SHA-256 |
| --- | --- |
| movement_libras.data | `97ebdaa6a9b28ab4a2cdd84b14f19a95a7456a46137c362b65a0669eca3c3c4d` |
| movement_libras.names | `1b0702f0b664c84e66a908f07a967e56f18aea2b8a85d40f44a0d1cc2fd18c35` |

The `.names` file remains in original Latin-1 encoding. The 360 source rows contain 45 alternating x/y coordinate pairs and a label from 15 movement classes. These are processed normalized hand-centroid trajectories from roughly seven-second recordings, not supplied live timestamps, image frames, complete signed-language sentences or raw neural signals. Four performers and two sessions are described, but per-row identities are absent.

## Boundaries before fitting

Group exact 90-coordinate rows and check label agreement within each group. Keep the first occurrence of each of 330 unique trajectories; the 30 extra copies do not cross the split. Use a single NumPy generator with seed73, class order1–15, classwise permutation: first floor(two-thirds) training, last four test, remainder validation. Counts220/50/60 and exact one-based source rows/duplicate groups are in author-results.json.

Whole trajectories define the split. A row-level result cannot assess unseen performers/sessions. Labels only stratify. Fixed transformation `2*x-1` applies to inputs and targets; positions0–43 predict1–44 causally. RMSE is square root of mean squared error over all transformed coordinates and positions, divided by2 to return to original units. No fitted normalization or test statistics enter training. This familiar dataset/split is reused for continuity across lessons, not new independent evidence or a controlled comparison against earlier different tasks/protocols.

The experiment declaration was saved in design before fitting. Three models share exact seed137 initial parameters, 4,946 parameters each, 160 full-batch Adam updates at0.003, and selection by minimum validation MSE with earliest exact ties. Model-specific differences are only dense causal softmax, a five-total-key window, or positive ELU+1 feature-kernel attention. All choose update160; no schedule extension or seed search follows. The complete architecture and dimensions are in author-calculations.py and manuscript§7. Unlike preceding GQA/MLA examples, this study uses **ordinary sinusoidal position addition**, three8-wide heads, and its own fitting protocol.

Persistence and six-parameter least-squares affine baselines use the same split. The affine map is fitted on training transitions only. Recorded test original-coordinate RMSE: persistence0.0264584944, affine0.0262221852, dense0.0232187454, window0.0246844795, kernel0.0246803332. Preserve all outcomes and the nearly equal window/kernel result. No timing or large-model language-quality claim follows.

## Actual executions and saved states

Environment: Python3.12.14, NumPy2.3.5, PyTorch2.14.0+cpu; one CPU thread and deterministic algorithms. Shared author runtime was used read-only; no dependencies installed. Complete study ran successfully once. Programs require local data and no network; installation commands in the manuscript are for the learner's own environment.

`author-calculations.py` contains all data handling, baselines, layers, training, selection, native full-prefix/incremental processing, float64 gradient comparison, forecast export and random-feature comparison. `author-results.json` contains environment, row boundaries, baseline fits, every training/validation update and final metrics. `forecast-models.json` contains the three selected checkpoints and full actual traces for worked and fresh examples. No model or hidden ground-truth forecast is fabricated.

The predeclared worked example is source77, prefix32, frame23 x→1−x. The fresh gated case is prefix27, frame19 y→1−y. These hypothetical edits do not create newly observed data or new correct next-point labels. Source77's actual future remains a target only for its original prefix. Full-prefix/incremental maximum transformed-coordinate errors are below2.4e−7; kernel quadratic-reference error below2.7e−7. Strictly earlier outputs are exactly unchanged for each edit. Float64 whole-network kernel evaluation-order checks give output error3.33e−16 and maximum parameter-gradient error3.55e−15. The inspected kernel denominator minimum is3.39313745499, far above its declared1e−9 floor.

`author-checks.py` reloads saved weights, reproduces the fresh original forecasts exactly, moves the edit to frame 25 inside the local window, checks all three final forecasts change and strictly earlier outputs remain unchanged, and executes the manuscript's displayed NumPy block as written. `fresh-controls.json` records those actual near-edit results and additional manual controls: sparse constant values; future zero-coefficient and inside-prefix projection changes; causal random-feature key edit; and the fresh nested m8→m64 error increase, 0.07823265999→0.13450358670. No fit is repeated for these controls. The displayed program returns `[2. 1. 2.]`.

At prefix32, numeric float32 attention payloads are6144/960/864bytes for dense/window/kernel; atprefix27 they are5184/960/864. These count actual K/V or S/z tensors only. Positions, metadata, weights, activations and allocator overhead are excluded. The training program materializes small dense masks and all kernel prefix states for autograd; constant streaming state does not imply constant training memory or a fast sparse kernel.

`random-feature-results.json` evaluates the selected dense model's actual head0 Q/K/V on prefix32, using already scaled Q/K, eight fixed Gaussian projection seeds and nested counts16/64/256. It saves every trial and its last weight/output row. Relative whole-head output error means are0.0432927808,0.0310981416,0.0170259743. Seed5 worsens16→64 and seed6 worsens64→256; do not remove them. Stabilization uses a query-specific common scalar and one scalar common to all keys, preserving the normalized sampled operator. There is no trained Performer or ORF task benchmark here.

`mechanism-calculations.py` and `mechanism-results.json` provide independent constructed fixtures: removed mass, sparse gathered/masked equality, causal graph reach, outer-product state and eviction, collisions, full versus prefix sequence projection, Nyström matrix calculation, Gaussian-marginal orthogonal rows, tile occupancy, fresh random-feature inputs and changed practice. The fixed-radius1d contrast is analytic, not a measured sample mean. These examples are explicitly constructed. Author-only answer arrays must remain hidden until the learner commits a prediction in phase two.

## Retention and next phase

Retain all packet files: manuscripts/specifications/design/provenance, original data/metadata, complete author programs and the required observed/calculated inputs/results. No new disposable source PDFs or scratch utilities were retained. Root owns exact-file checkpoint hashes and phase status. Phase two should derive small semantic on-demand assets, provide complete reproduction downloads, implement the specified visuals/labs, and perform independent/runtime/rendered/accessibility/integration checks. Do not eagerly import full training histories, all source records or author-only answers into the initial lesson bundle.
