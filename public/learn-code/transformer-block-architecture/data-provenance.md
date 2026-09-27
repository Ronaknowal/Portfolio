# Transformer Block Architecture — data and computation provenance

Research/write phase, 13 September 2026. These files are pending teaching/verification inputs, not disposable scratch. No browser training, large pretrained model, synthetic benchmark curve or GPU timing is part of this packet.

## Real data and redistribution

Dataset: **Libras Movement**, Daniel Baptista Dias, Sarajane Marques Peres and Helton Hideraldo Bíscaro; University of São Paulo. Citation: Dias, D., Peres, S., & Bíscaro, H. (2009), UCI Machine Learning Repository, [DOI 10.24432/C5GC82](https://doi.org/10.24432/C5GC82). The [canonical UCI 181 page](https://archive.ics.uci.edu/dataset/181/libras%2Bmovement) explicitly gives [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) rights. Its full data-description page and the original `.names` metadata were inspected during the immediately preceding self-attention packet; that checked acquisition evidence is reused here, not represented as a new download. The original metadata contains no conflicting commercial-use restriction.

The official source archive was [libras+movement.zip](https://archive.ics.uci.edu/static/public/181/libras+movement.zip),164,761 bytes, SHA-256 `486f22f63919002a991be4042d7c7489a56cbd5f40f2930a24f80744c93e7105`, obtained 13 September 2026. The two required inputs were copied byte-for-byte from the preceding topic-owned packet; its source/rights record remains frozen. No archive, videos or source variants were retained here.

| File | SHA-256 | Meaning |
|---|---|---|
| `movement_libras.data` | `97ebdaa6a9b28ab4a2cdd84b14f19a95a7456a46137c362b65a0669eca3c3c4d` | Original 360 records,90 coordinates+class |
| `movement_libras.names` | `1b0702f0b664c84e66a908f07a967e56f18aea2b8a85d40f44a0d1cc2fd18c35` | Original metadata, Latin-1 accented source names preserved |

Records contain 45 ordered x/y hand-centroid samples from approximately seven-second movement videos, with normalized coordinates 0–1. The columns alternate x1,y1,…,x45,y45,class. There are 15 movement-type classes,24 raw rows each. They do not encode complete signs/sentences or full body pose. Collection describes four performers and two sessions, but reliable per-record performer/session IDs are unavailable; no identity split is invented from row ordering.

## Independent boundary and transformation

Exactly comparing the 90 coordinates identifies 330 unique trajectories and 30 additional repeated rows; repeated feature groups all have consistent class labels. Keep each unique trajectory's first source occurrence. Preserve the raw source and exact duplicate groups as evidence. Within each class, use one NumPy generator seed 73 to shuffle unique source rows, take the first two-thirds for training, the last 4 for test and the middle remainder for validation. Totals 220/50/60; each test class has 4 rows. The entire fixed row lists and duplicate groups are saved in `author-results.json.data` using one-based source IDs. No trajectory's points are split across folds; no exact duplicate spans the splits after deduplication. Absence of performer IDs still limits the claim to this fixed row-level study.

The same input/split is reused from self-attention for learning continuity, not to claim independent datasets across chapters. Its earlier model results are not merged with the present fits to attribute a gain to one architectural ingredient.

Each point uses fixed coordinate transformation `2*x−1`. Append a deterministic tag `s_t=2t/44−1` for t0…44 before the learned 3-to-24 stem. Tags are normalized position indices, not measured elapsed time. No fitted scaling or test-derived preprocessing. If padding is appended, retain the original 45 tags and assign tag 0 to padded records. A joint permutation moves the original tag with its point; a time-reversal intervention reverses coordinates against unchanged tags. These are distinct input changes.

## Declared native study and evidence

The design's experiment declaration was written before fitting. Two blocks, width 24,2 heads of width 12, FFN width 48, exact GELU, biased maps, affine LN epsilon 1e−5. Both pre/post models include a final affine LayerNorm before mean pooling and a24-to-15 classifier. Only within-block normalization placement changes, giving 10,263 parameters in each. The shared final-normalizer convention is deliberate control, not an original 2017 reproduction. Seeded models have identical parameter tensors before their different forward rules.

Six declared fits: placement pre/post × seeds 101/102/103; Adam .003, no weight decay, no dropout,180 full-batch epochs. Select validation macro F1, then lower validation CE, then earlier exact tie. Each test score is computed after its checkpoint selection. No retuning followed the results. All histories, checkpoints, final split scores and parameter counts appear in `author-results.json`. Observed test-correct counts are pre 49/49/51 and post 52/51/47 out 60. Cross entropy is natural-log nats/example. Macro F1 includes all 15 classes; zero-division cases map to0 in the declared metric.

Source row 77/class 4 is the predeclared first test example of class 4. Both seed 101 models misclassify it (pre class 7,post class 3), and both failures remain. `block-models.json` contains the two selected state dictionaries, input/time tags, probabilities/logits, baseline stage tensors/attention matrices, confusion matrices, and the actual paired-permutation/reversal/reflection/padding interventions. Dense weights use PyTorch output-by-input storage. MHA uses packed Q/K/V maps and biases in that order. The first true time/coordinate edit changes predictions; a joint-record permutation and correctly masked padding are numerical nulls. Arbitrary future edits must compute the actual function, not interpolate saved outputs.

Execution environment: Python 3.12.14, NumPy 2.3.5, PyTorch 2.14.0+cpu, scikit-learn 1.9.1. PyTorch one CPU thread, deterministic algorithms, no GPU or dependency installations. The full source program executed successfully. Runtime values are not a speed benchmark and no claim of a large-paper replication is made.

## Exact and independent calculations

`author-calculations.py` also saves a two-token exact block fixture, zero-branch controls, changed-input stage tensors, one-vector normalizer/sum-probe contrast, and a same-parameter comparison to the installed TransformerEncoderLayer for both placements (maximum absolute differences 0 under the stated float64 execution). Initialized depth 1/4/12 diagnostics use block seed 53, independent input/probe generator 97, d8/h2/f16, a final shared affine-free LN and a unit-length random output probe. Gradients are with respect to carried states; the corresponding all-ones-probe null is retained. These are initialization measurements, not trained model quality or singular-value spectra.

`additional-calculations.py` uses independent NumPy formulas to compare the LN analytic derivative with central finite differences (step 1e−5, max error 8.37e−12), checks the all-ones null, computes a small-spread amplification example, direct-FFN position independence, changed practice vectors/counts, entropy denominator and derived MAC values. It writes `additional-fixtures.json` and performs no fit or network access. The manuscript's three displayed Python programs were extracted and executed; all reported shapes/counts/gradients match.

Keep exact synthetic teaching fixtures, measured real-data results and derived counts labeled separately. No old publication's manually entered activation/gradient/speed plots are preserved as empirical data. No new labels are assigned to edited real trajectories. The primary data license/attribution must accompany redistribution or derived displays.

## Phase-two retention and bounded delivery

Required packet files: manuscript/specification/design/provenance, the two original data files, both calculation programs, both results JSON files and `block-models.json`. The retained model JSON includes baseline traces for parity evidence; phase two can derive smaller actual-weight assets and load only the selected model on demand, offering the whole reproduction package as a separate download. Do not make the browser eagerly fetch all six training histories or execute the training loop. This conversion must preserve actual weights and validation evidence.

No disposable scratch was created. Runtime implementation, browser/accessibility checks, artifact packaging, formal independent phase-two review and publication remain deferred. Any concrete implementation correction should update its affected manuscript/specification/source evidence and phase checkpoint through the root owner; unchanged numerical campaigns need not be repeated.
