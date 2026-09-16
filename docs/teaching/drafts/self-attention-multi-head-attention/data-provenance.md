# Data and calculation provenance

Research/write phase, 13 September 2026. This packet retains small required inputs and model parameters because phase two must reproduce genuine edited-input inference. They are not scratch outputs to delete before implementation.

## Source and rights

Dataset: **Libras Movement**, Daniel Baptista Dias, Sarajane Marques Peres and Helton Hideraldo Bíscaro; donor University of São Paulo. Dataset citation: Dias, D., Peres, S., & Bíscaro, H. (2009), UCI Machine Learning Repository, [DOI 10.24432/C5GC82](https://doi.org/10.24432/C5GC82).

The [current primary dataset page](https://archive.ics.uci.edu/dataset/181/libras%2Bmovement), inspected in full, explicitly licenses sharing and adaptation under [Creative Commons Attribution 4.0](https://creativecommons.org/licenses/by/4.0/). The official archive's original `movement_libras.names` was read in full, including its source, collection, attribute, class and prior-use sections. It contains no conflicting commercial-use restriction. The named study's historical nearest-neighbor figures are not our benchmark and are not reproduced as comparable results.

Fetched [the official 164,761-byte archive](https://archive.ics.uci.edu/static/public/181/libras+movement.zip) on 13 September 2026. Its SHA-256 was `486f22f63919002a991be4042d7c7489a56cbd5f40f2930a24f80744c93e7105`. Only the complete data file and original metadata needed by this lesson were retained. No archive, variant subsets or raw videos were retained.

| Retained source input | SHA-256 | Handling |
| --- | --- | --- |
| `movement_libras.data` | `97ebdaa6a9b28ab4a2cdd84b14f19a95a7456a46137c362b65a0669eca3c3c4d` | Byte-original 360-row file |
| `movement_libras.names` | `1b0702f0b664c84e66a908f07a967e56f18aea2b8a85d40f44a0d1cc2fd18c35` | Byte-original metadata; read as Latin-1; preserve original accented names |

Source coordinates are normalized two-dimensional hand centroids sampled at 45 ordered positions from approximately seven-second movement videos. The file layout is x1,y1,x2,y2,...,x45,y45,class. The 15 classes are movement types; the data does not contain complete linguistic signs or sentence translations. It is not appropriate to assign this model a sign-language translation capability. The four performer/two-session collection description does not supply an individual/session ID for each row. Do not infer those IDs from row order.

## Duplicate boundary and fixed split

The raw file has 360 rows and 90 finite features per row; every class has 24 rows. Comparing all 90 coordinates exactly yields 330 unique trajectories. Thirty additional rows are exact copies. Each repeated feature group has the same label. Keep its first source occurrence for the experiment; keep all source bytes as evidence. The complete one-based duplicate groups and train/validation/test row IDs are in `author-results.json` under `data`.

Within each source class, permute the retained unique row IDs using one NumPy generator with seed 73. The first two thirds are training, the last four are test, and the remaining middle rows are validation. Total counts are 220/50/60. Test class counts are exactly four each. Point-level splitting would leak the same trajectory into multiple sets and is not used. Exact deduplication avoids that particular leakage route, but lack of performer/session IDs limits the interpretation to this fixed row-level study.

Use fixed `2*x - 1` coordinates, with no learned preprocessing. No label, probability or selected checkpoint is inferred from a fabricated trajectory. The interactive original is source row 77/class 4, chosen by the declared first-test-example-of-class-4 rule before model fitting; its clockwise-arc misclassification is preserved.

## Actual native computation and artifacts

`author-calculations.py` was executed once for the complete fixed 12-fit campaign. It uses the full 220-row training batch, Adam at .01, no weight decay, 200 epochs and seeds 101/102/103 for each of four declared models. Each checkpoint is selected by validation macro F1, then lower validation cross entropy, then earlier epoch. Test examples are scored only after selection for each declared fit. The seed-101/two-head model was predeclared for visualization, not chosen as a favorable seed. `author-results.json` records versions, all model metrics, training/validation histories, exact split and hand fixtures. The source collection and experiment limitations are in the learner manuscript once at their relevant home.

Recorded runtime: Python 3.12.14, NumPy 2.3.5, PyTorch 2.14.0+cpu, scikit-learn 1.9.1. One PyTorch CPU thread, deterministic algorithms, no GPU, no installation into the shared runtime. Results are local observations, not a hardware benchmark or reproduction of a large-paper score.

`attention-model.json` contains the actual selected seed-101/two-head state dictionary, its source trajectory, original attention matrices, probabilities, exact edit/padding/reversal probes and test confusion matrix. Linear weights use PyTorch's output-by-input storage. It has 2,751 learned scalar parameters. The browser should load this small asset only for the relevant lesson/investigation and recompute the model for user edits, not interpolate among saved outputs or train a new network.

`additional-calculations.py` computes changed practice fixtures and the all-points-versus-one-point duplication controls from this frozen model. It writes `additional-fixtures.json`; it performs no model fitting. All numeric charts must identify whether they show the exact hand fixture, derived operation/memory counts or these actual fitted-model observations.

The independent hand fixtures verify causal renormalization, a score-translation null, a same-values counterexample, analytic attention gradients, rectangular decode-mask alignment and same-parameter agreement with the installed `MultiheadAttention`. Those are bounded author calculations supporting the text, not a formal phase-two integration/test campaign.

## Prior candidate and boundary

Japanese Vowels was considered and rejected before any of its data was retained or modeled. Its current UCI page says CC BY 4.0, but the [original donor metadata](https://kdd.ics.uci.edu/databases/JapaneseVowels/JapaneseVowels.data.html) requires donor permission for commercial use. No donor-authorized relicensing was established. This specific source conflict motivated choosing Libras; it does not assert a restriction on other UCI datasets. The evidence was sent to the neighboring sequence author to prevent inconsistent reuse.

## Phase-two retention and delivery

All files in this packet are pending teaching inputs. Keep the two source files, full programs, results and small model while implementing the lesson. Publish the source attribution/license with any redistributed data or derived visual. Give learners the complete offline reproduction package; replace draft-relative links with the actual download routes. The website need not eagerly ship the 360-row training set or run the native training program. Rendering/accessibility, packaging and browser/runtime verification remain implementation work.
