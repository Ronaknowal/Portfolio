# Positional Encodings — data and calculation provenance

Research/write phase, 13 September 2026. These are required pending teaching inputs. No runtime implementation, browser training or large pretrained model acquisition occurred.

## Original data and rights

Dataset: Libras Movement, Daniel Baptista Dias, Sarajane Marques Peres and Helton Hideraldo Bíscaro, University of São Paulo. Cite Dias, D., Peres, S., & Bíscaro, H. (2009), UCI Machine Learning Repository, [DOI 10.24432/C5GC82](https://doi.org/10.24432/C5GC82). The [canonical UCI 181 page](https://archive.ics.uci.edu/dataset/181/libras%2Bmovement) specifies [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Original `.names` metadata and source description were fully inspected in the immediately preceding authoring chain. There is no conflicting restriction in those original Libras metadata. Reuse that source-bound rights evidence instead of presenting a repeated acquisition as new research.

The source archive was [libras+movement.zip](https://archive.ics.uci.edu/static/public/181/libras+movement.zip), 164,761 bytes, SHA-256 `486f22f63919002a991be4042d7c7489a56cbd5f40f2930a24f80744c93e7105`, obtained 13 September 2026. Copy the two required original files byte-for-byte from the frozen Transformer Block packet. The discarded archive itself is not needed here.

| File | SHA-256 |
|---|---|
|`movement_libras.data`|`97ebdaa6a9b28ab4a2cdd84b14f19a95a7456a46137c362b65a0669eca3c3c4d`|
|`movement_libras.names`|`1b0702f0b664c84e66a908f07a967e56f18aea2b8a85d40f44a0d1cc2fd18c35`|

The `.names` file preserves its original Latin-1 accented names. Data contain 360 rows, 90 alternating x/y coordinates and one class. Each trajectory is 45 hand-centroid samples from approximately seven-second movements; normalized coordinates 0–1, 15 movement classes. Full signed-language interpretation is not supplied by these features. Four performers and two recording sessions are described, but reliable per-row IDs are absent. Do not invent a performer/session split from record ordering.

## Input boundary and protocol

Compare all 90 coordinates exactly: 330 distinct trajectories and 30 additional repeated rows; each feature-duplicate group has one consistent class label. Retain first occurrence per group. Using one NumPy seed 73 generator, independently permute retained rows within each class in class order; assign first floor(two-thirds) to training, last 4 to test, middle remainder to validation. Totals 220/50/60. Author JSON retains exact one-based source row lists and all duplicate groups. A trajectory or its exact duplicate never crosses the split. This is still a row-level diagnostic without a new-performer claim.

Fixed coordinate transformation `2*x-1`; no fitted scale and no test-derived input statistics. Logical position IDs 0–44 are ordinal sample positions, not exact seconds. The positional comparison uses **one** block and no explicit scalar time tag, unlike the preceding two-block normalization study. Do not combine their scores as an isolated architectural head-to-head. Reusing the same real task makes the examples coherent but does not create independent benchmark datasets.

The experiment declaration was saved in `design.md` before fitting. Five modes none/sinusoidal/learned/RoPE/symmetric bidirectional ALiBi; shared 2-to-24 stem, pre-norm one-block 2 heads of width 12, FFN 48 GELU, final norm, valid mean pool and 15-class output. Affine LN epsilon 1e−5, biased maps, no dropout. Shared seed 101 tensors asserted identical before mode-specific computation. Learned 45 × 24 table initialized separately with generator 303 and std 0.02; 6,447 parameters versus 5,367 in the other modes. Sinusoidal/RoPE base 10000. ALiBi original 2-head slopes 1/16, 1/256 with symmetric absolute-distance encoder adaptation; no causal mask in this classifier.

Each fit 180 full-batch Adam updates at 0.003, no weight decay. Select validation macro-F1, lower CE, earlier exact tie; final test only after selecting that model. Macro-F 1 includes all 15 classes with zero-division contribution 0. Natural-log CE units nats/record. No further hyperparameter/seed campaign followed the results. The one declared seed supports this bounded mechanism study, not an architecture ranking.

Recorded selected epochs 107/154/108/106/180, test correct 33/42/40/40/39 of 60 in declared mode order. All split metrics and validation histories are in `author-results.json`; full program `author-calculations.py` produced them. Environment Python 3.12.14, NumPy 2.3.5, PyTorch 2.14.0+cpu, sklearn 1.9.1, one CPU thread, deterministic algorithms. No dependency installation, GPU or internet access in the reproduction program.

## Real displayed input and interventions

Source row 77/class 4 was predeclared as first test row of class 4. Every selected model misclassifies it; preserve those failures. `position-models.json` contains all five actual weights, raw input/position IDs, actual raw/rotated Q/K/value/score/attention/pooling traces, baseline outputs, intervention outputs and 60-record confusion matrices. Packed linear output layout is Q24, K24, V24, then each splits into two 12-dimensional heads. Dense tensors use PyTorch output-by-input storage. Histories remain author/reproduction evidence, not eager browser assets.

Jointly reverse points and IDs: mathematical permutation null, observed max 1.55e−6 in float32. Reverse points while keeping IDs 0–44: none and symmetric ALiBi remain invariant; sinusoidal/learned/RoPE change logits. Reflect frame 23 x→1−x: actual changed-input output computed for every model, not an interpolation of saved outcomes. Append 5(.75,.75) pads with safe ID 0 while retaining original IDs, mask keys and pool: baseline recovered within 1.5e−6; unmasked variants are also recorded. No edited trajectory is assigned a new ground-truth class or included in accuracy counts.

The symmetric ALiBi reversal result follows its encoder distance matrix plus shared rowwise blocks plus mean pooling. It is not a statement that original causal ALiBi language models cannot distinguish direction. The geometry, actual numerical null and content-dependent change are all needed to explain the finding.

## Independent exact fixtures and displayed program

`mechanism-calculations.py` uses NumPy formulas independently of the PyTorch classifier. It calculates sinusoidal tables, rotary norms/dots/common shifts, a constant-content Toeplitz matrix and changed-content counterexample, nonmonotone cosine fixture, ALiBi content/distance odds, T5 signed buckets, fixed-content full/cached attention, wrong query offset and stale/new-frequency contrast, PI/base/YaRN-paper-ramp frequencies and changed-practice values. Results are in `mechanism-fixtures.json`. These are constructed exact teaching inputs and formula-derived outputs, not learned activations or empirical benchmark curves.

The manuscript's complete standalone NumPy program was extracted and run as displayed. Its strict learned lookup, sinusoidal vector, non-power-of-two slopes, full/cached RoPE and ALiBi outputs, and wrong-angle example matched the reported output. Bounded additional checks and final author reread are recorded in design. A later implementation needs browser parity and visual/accessibility checks; no screenshot or formal phase-two acceptance is claimed here.

## Retention and delivery

Retain manuscript, specifications, design, this provenance, both raw data files, both full programs, both result JSON files and the real model JSON: 11 files. They are pending implementation inputs, not disposable scratch. No new disposable scratch was created. Phase two can derive compact per-model weight assets from the evidence, load only the selected model, and offer the full program/data package separately. Preserve license/attribution and actual data transformations. The browser should evaluate ≤50 points for one selected small model, never train or replay all fits. Root owns central phase/hash updates.
