# Libras Movement — retained data and experimental roles

Source: [UCI Libras Movement, dataset181](https://archive.ics.uci.edu/dataset/181/libras%2Bmovement), DOI[10.24432/C5GC82](https://doi.org/10.24432/C5GC82). Credit Daniel Dias, Sarajane Peres and Helton Bíscaro, University of São Paulo, for the2009 dataset. UCI declares [Creative Commons Attribution4.0](https://creativecommons.org/licenses/by/4.0/). Preserve attribution and identify transformations in the learner download and any runtime export.

Research/reuse date:13September2026. The author read the current UCI page and the full original `movement_libras.names` file (Latin1 text). The same increment's self-attention author downloaded the official [archive](https://archive.ics.uci.edu/static/public/181/libras+movement.zip), read its metadata and permitted byte-original reuse. These two source files were copied unchanged from that author's retained packet; this author did not claim a second network acquisition. The source names file has no additional commercial-use restriction.

| File | SHA256 |
| --- | --- |
| Official archive, acquisition author's recorded digest; not retained again here | 486f22f63919002a991be4042d7c7489a56cbd5f40f2930a24f80744c93e7105 |
| movement_libras.data, retained byte-original | 97ebdaa6a9b28ab4a2cdd84b14f19a95a7456a46137c362b65a0669eca3c3c4d |
| movement_libras.names, retained byte-original | 1b0702f0b664c84e66a908f07a967e56f18aea2b8a85d40f44a0d1cc2fd18c35 |

## Meaning and limits

There are360 rows,24 per original movement class. Each has90 coordinate features ordered x1,y1,...,x45,y45, followed by a label1–15. The metadata describes uniformly selecting45 frames from time-normalized videos of about7seconds, using the hand centroid to create a two-dimensional trajectory in unit space. Four people contributed in two sessions. No row-level performer/session identifier is included. This exercise therefore cannot measure held-out-person/session transfer, infer exact temporal intervals or physical velocity, or represent complete sign-language meaning.

Display classes follow the names file:1curved swing;2horizontal swing;3vertical swing;4anti-clockwise arc;5clockwise arc;6circle;7horizontal straight-line;8vertical straight-line;9horizontal zigzag;10vertical zigzag;11horizontal wavy;12vertical wavy;13face-up curve;14face-down curve;15tremble. Keep original terminology attributed to the dataset. Program class indices subtract1.

## Exact preparation

`latent_trajectory_classifier.py::prepare` loads the source values as float64. It groups exact coordinate tuples, confirms all duplicate labels agree, and retains first source occurrence. There are330 unique trajectories and30 duplicate copies. Complete one-based duplicate groups and retained source IDs are in `trajectory-results.json`.

Within each class in ascending order, NumPy default_rng(73) permutes retained source indices. First floor(2n/3) fit; last4 test; the remaining middle rows validate. Totals220/50/60, with four test trajectories per class. All roles are disjoint and exhaust the330 retained rows. This is a declared row-level partition, not evidence that performers are statistically independent across roles.

Raw coordinates are transformed by the fixed rule2x−1 and converted to float32, with an attached float32 ordinal tag linspace(−1,1,45). No fitted preprocessing uses validation/test data. Mean baseline inputs are the two normalized coordinate means; ordered baseline inputs are the ninety normalized coordinates. Exact copies are removed before role creation. The byte-original source files retain all360 rows for provenance; experimental roles use330.

## Computations and retained outputs

The complete CPU program trains C1 logistic mean and ordered-coordinate baselines; and two-read width24 latent models with N1/4, seeds11/29, Adam learning rate.005,80full-batch epochs. Validation cross-entropy selects each neural epoch before test evaluation. The tested environment was Python3.12.14, NumPy2.3.5, PyTorch2.14.0+cpu, scikit-learn1.9.1. No learned model or hyperparameter was tuned after viewing these results.

`trajectory-results.json` records roles, duplicate groups, all four histories, selected epochs, confusion matrices and metrics. `small-fits.npz` retains selected state dictionaries and all source-row logits, plus logistic weights. It contains no pickled objects; load with NumPy's default allow_pickle=False. These are small original educational fits, not research-model checkpoints.

`investigation-checks.json` records the chosen validation source row7, exact full/first-point/moved-point/retimed outputs, paired-record/padding nulls, derivative/attention/affine calculations and role checks. It identifies the bounded fixture-selection rule. Edited trajectories are hypothetical, so their original labels do not become certified edited-input labels.

Source files and generated retained arrays are necessary for the pending content-first handoff. Runtime export, licensing presentation, exact forward parity, desktop/mobile/accessibility/performance checks and independent lesson review remain phase two.
