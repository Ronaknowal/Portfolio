# Offline data and measured-result provenance

Retrieved/reviewed 13 September 2026. Dataset: **Wine**, Stefan Aeberhard and M. Forina (1992), UCI Machine Learning Repository, DOI [10.24432/C5PC7J](https://doi.org/10.24432/C5PC7J), [dataset page](https://archive.ics.uci.edu/dataset/109/wine). The page describes chemical analysis of wines from three cultivars grown in the same Italian region; the classification target is cultivar. This dataset has 178 examples, 13 real-valued features and class counts59/71/48.

The UCI page explicitly licenses the data under [Creative Commons Attribution 4.0 International](https://creativecommons.org/licenses/by/4.0/). Credit: Aeberhard and Forina / UCI, with the DOI above. This packet adapts a copy supplied by `sklearn.datasets.load_wine` in installed scikit-learn1.9.1, exported locally on13 September2026; it did not scrape a new upstream file. All numeric feature values match that bundled data exactly when reloaded. We add zero-based original row IDs, supply a header and map original UCI class codes1–3 to0–2. Row IDs and targets are not model features. The altered training-label experiment is explicitly a teaching intervention and leaves the source CSV unchanged.

`wine.csv` SHA-256: `68386101f5ee41ef2a03d50bb8fe6dc5d4ac86aac31b6e987616be00a8c9c728`. Column order after row ID/target: alcohol, malic_acid, ash, alcalinity_of_ash, magnesium, total_phenols, flavanoids, nonflavanoid_phenols, proanthocyanins, color_intensity, hue, od280/od315_of_diluted_wines, proline. Names preserve scikit-learn's supplied spellings. Do not infer a common measurement unit for all thirteen columns.

`split.json` is a deterministic partition created with NumPy2.3.5 `default_rng(2026)`: independently permute each class's source row IDs in class order0,1,2, take24 training and next12 validation, reserve the rest. Training72 and validation36 are class balanced; test70 counts are23/35/12. The tiny12 contains the first4 selected training IDs of each class. Exact IDs are retained because a seed alone is not a data-version-independent membership identifier.

The full CSV has no reliable deployment grouping/time keys. A random row split establishes only the scoped historical exercise. The70 reserved test rows are present for transparent partition provenance but never evaluated by these programs. The fitted mean and population standard deviation are computed from the72 training feature rows; the same transform is applied to all feature rows and reused by the tiny diagnostic. Full-data models receive the13 standardized features.

## What each artifact is

| Artifact | Meaning and generation |
| --- | --- |
| `experiment-protocol.md` | Fixed settings, seeds, budgets, split and intended comparisons, written before first neural training execution |
| `wine_diagnostics.py` / `wine-results.json` | Actual float64 CPU runs for tiny correct/omitted update plus clean/shuffled labels under seeds3,7,19; every evaluated update, fitted scale, activation quantiles and environment retained |
| `checkpoint_replay.py` / `checkpoint-results.json` | Actual uninterrupted/resumed CPU trace through12 updates, saved after5; exact restore plus one omitted state category per comparison |
| `calculations.py` / `calculation-results.json` | Exact rational scalar fixtures, autograd-versus-finite-difference comparison, four-way actual BatchNorm mode probes, independent-input and null checks |
| `author-verification.json` | Final bounded artifact, numerical, source and displayed-program checks with exact input/program hashes |

The neural environment actually used is Python3.12.14, NumPy2.3.5, PyTorch2.14.0+cpu; scikit-learn1.9.1 is used only for the offline export/verification, not needed to execute the supplied fits. `wine-results.json.environment` records Windows/processor information, float64, deviceCPU, one Torch thread and deterministic algorithms. No timing or throughput claim is made.

The checkpoint intentionally omits one state category at a time. In the “optimizer buffers” treatment, current learning rate and parameter-group settings are restored and only momentum buffers are removed. This keeps a momentum-history question from also changing the current learning rate. The scheduler is constructed before optimizer state loading. Global Torch RNG is restored after all constructors; a separate Torch generator owns permutations. Saving is at a completed-update/cleared-gradient boundary. These simple CPU results do not establish a multiworker, distributed or cross-device resume protocol.

Both example programs write their result JSON beside the program when run. Preserve the supplied output before changing settings. No disk checkpoint binary, remote model, data downloader or external experiment-tracking service is required; serialization in the replay probe passes through an in-memory buffer. No full paper benchmark, GPU training, browser training or website visual implementation was performed.
