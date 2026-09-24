# AutoML & NAS — observed input and calculation provenance

Content-only evidence record, 12 September 2026. This packet is retained for implementation handoff. Browser rendering, native optional AutoML frameworks, formal implementation review, and publication remain pending.

## Original observations

- Dataset: **Banknote Authentication**, attributed by UCI to Volker Lohweg (2012); the repository records donation on 15 April 2013.
- Landing page and documentation inspected: [UCI dataset 267](https://archive.ics.uci.edu/dataset/267/banknote+authentication).
- Persistent identifier: [10.24432/C55P57](https://doi.org/10.24432/C55P57).
- Download: [official UCI ZIP](https://archive.ics.uci.edu/static/public/267/banknote+authentication.zip), retrieved 12 September 2026.
- License: [Creative Commons Attribution 4.0](https://creativecommons.org/licenses/by/4.0/), as stated by the UCI dataset page. Attribute the source when distributing the file or derived teaching inputs.
- ZIP SHA-256: `1e2acd9a2085fadf3d8145c12d3d22af853320d52294a6590c2eaf75fdc05227`.
- ZIP member: `data_banknote_authentication.txt`. Retained as **`banknote-data.csv`**, with unchanged bytes and no header added.
- Retained file size: 46,400 bytes. SHA-256: `d0539aaed2139ba7a587b3e34fb345ce503ff7d5d33dbf9912d8e195ce425cb9`.
- Original rows: 1,372. Original numeric class counts: 762 class 0, 610 class 1. No row or column was removed.

The four columns are supplied numerical descriptors: variance, skewness, kurtosis (source spelling “curtosis”), and entropy of image-derived measurements. The documentation describes images from genuine/forged banknote-like specimens, image dimensions of 400×400 pixels and a wavelet feature extraction step. This packet retains the supplied descriptors, not images. Descriptor physical units, complete extraction details, specimen identifiers and capture-session identifiers are not available in the inspected record. The numeric 0/1 mapping to genuine/forged was not established; the manuscript uses the original numeric labels.

This source is small and public, not a representative operational authentication benchmark. Its age and missing specimen/capture information constrain external conclusions. Keeping exact feature copies together addresses one identifiable split problem without asserting that each unique feature vector is a distinct independent physical object.

## Duplicate groups and fixed data roles

`np.unique(x, axis=0, return_index=True, return_inverse=True, return_counts=True)` creates groups in lexicographic order of the four feature columns. There are **1,348 unique vectors**, **11 groups with more than one row**, and **24 additional repeated rows**. All rows in each repeated group have the same class. Every original row is retained; groups, rather than rows, are partitioned. This is conservative blocking of identical observed feature vectors, not recovered physical identity.

One class label per unique vector is used for stratification. Seed 71 splits 900 development groups from 448 remaining groups. Seed 72 splits the remainder into 200 inspection groups and 248 reserved groups. Row arrays use original file order through `np.flatnonzero(np.isin(group, selected_groups))`.

| Role | Groups | Rows | Class 1 | Class 0 |
| --- | ---: | ---: | ---: | ---: |
| Development | 900 | 919 | 407 | 512 |
| Inspection | 200 | 205 | 91 | 114 |
| Reserved | 248 | 248 | 112 | 136 |

All roles are disjoint in rows and exact-feature groups and collectively cover all 1,372 source rows. The reserved set receives **no model predictions**. Its counts describe the fixed partition, not model selection evidence.

Within development groups, `StratifiedKFold(3, shuffle=True, random_state=73)` supplies fitting/validation group IDs that are mapped back to every associated original row. Validation row counts are 314, 300, 305; fitting counts are 605, 619, 614. All candidates use these same folds. Every development row receives one out-of-fold prediction per candidate, aligned to `roles.development_ids`. Any learned scaler is fitted inside the pipeline on the relevant fitting rows.

## Declared candidates and fitting convention

Registry order is part of the tie rule:

1. Logistic, raw, C = 0.1.
2. Logistic, raw, C = 1.
3. Logistic, standard scaling, C = 0.1.
4. Logistic, standard scaling, C = 1 — **predeclared final baseline**.
5. Decision tree, maximum depth 2.
6. Decision tree, maximum depth 5.
7. Standardized nearest neighbors, k = 3.
8. Standardized nearest neighbors, k = 9.
9. Standardized tanh MLP, hidden width 8.
10. Standardized tanh MLP, hidden width 16.
11. Standardized tanh MLP, hidden widths 8 and 8.

Logistic fitting: `solver='lbfgs'`, `max_iter=1000`, `tol=1e-8`, the declared C. Tree fitting: declared depth and `random_state=74`. Neighbor fitting: declared k and scikit-learn defaults otherwise. MLP fitting: `activation='tanh'`, `solver='lbfgs'`, `alpha=.01`, `max_iter=1000`, `tol=1e-7`, `random_state=74`, declared widths. StandardScaler uses default centering and population-variance scaling convention. Its fitted statistics are fold-local. The MLP regularization follows this installed estimator's objective convention; its alpha is not silently identified with the separately normalized lambda in the earlier hand-derived regularization lesson.

Candidate selection maximizes the arithmetic mean of the three fold accuracies. Pooled OOF accuracy is retained for interpretation but does not select the winner. Exact mean ties choose the first original registry entry. Each fold uses a fresh cloned estimator. Native thread pools are limited to one thread for the calculation.

After all 33 fold fits, refit only the selected candidate and the predeclared standardized logistic C = 1 on all development rows. Predict only those two models on inspection. This adds two fits: **35 estimator fits total**. There was one fixed native calculation campaign, with no outcome-driven parameter changes or additional seed sweep. No native fit was rerun for manuscript polishing, visual contracts, or pure fixture verification.

## Retained outcomes

The selected candidate is `mlp-tanh-16`, with mean fold accuracy 1.0. Each of the three folds has zero errors. The remaining exact candidate means and folds are retained in `calculated-inputs.json`; manuscript values are rounded to six decimals for display.

`neighbors-standard-k3`, `mlp-tanh-8`, and `mlp-tanh-8x8` each misclassify only source-row index 349, true 0 and predicted 1. Their mean fold accuracy is 0.9989071038251366 and pooled accuracy is 918/919. Their shared mistake should not be interpreted as three independent sources of ensemble benefit.

Inspection results:

| Model | Correct / rows | Accuracy | Confusion matrix: true rows, predicted columns, order 0/1 |
| --- | ---: | ---: | --- |
| Selected MLP width 16 | 205 / 205 | 1 | `[[114,0],[0,91]]` |
| Predeclared standardized logistic C = 1 | 202 / 205 | 0.9853658536585366 | `[[111,3],[0,91]]` |
| Always predict development majority 0 | 114 / 205 | 0.5560975609756098 | Contextual deterministic rule; no estimator fit. |

The baseline errors are source indices 107, 195, 345. Source arrays use **zero-based** row IDs; learner-facing original file line numbers add one, giving 108, 196, 346. The out-of-fold shared-error index 349 is file line 350. Source file order has not been sorted or deduplicated.

The retained calculation recorded no fitting warnings. No model objects, fitted probabilities, or fitted weight arrays were exported. Visuals may inspect saved class predictions and input features; they may not invent continuous boundaries or probabilities. Perfect finite inspection accuracy supports only this particular declared assessment, with the documented grouping limitation and reserve still unused.

The seed-75 candidate order `[7,10,6,9,3,2,1,4,8,0,5]` is a predetermined random reveal order for the already calculated finite table. The best-so-far curve is a replay, not an independently executed search benchmark or a cost comparison against Bayesian optimization. Prefix tie decisions still use registry order.

## Constructed inputs and author verification

`constructed` in `calculated-inputs.json` is separate from observed data. It contains exact conditional counts, Gaussian expected-improvement examples, operation softmax/gradient values, a discretization counterexample, scalar bilevel gradients, nine declared fidelity curves, a three-configuration/two-task portfolio matrix, and hypothetical deployment points. Additional author-check fixtures contain the changed practice calculations and the activation-code kernel. These are created for instruction and carry no empirical performance claim.

Pure verification can recompute these values without invoking `calculate()`, which performs the native fits. The retained author program's `__main__` intentionally reproduces the full study; a finisher should run it only when a consequential change or required execution check justifies the cost. Do not repeatedly rerun it because a prose paragraph changed.

Current author environment: Python 3.12.14, NumPy 2.3.5, SciPy 1.18.1, scikit-learn 1.9.1. The shared author runtime was used read-only with `-B`; no package installation or runtime mutation occurred. Wall duration was observed operationally but is not retained as an educational timing benchmark.

The displayed main teaching program reproduces the same roles, candidate constructors and fitting/selection convention. Its parsed definitions were compared with the retained author program without another fitting campaign. Exact displayed-script execution remains a phase-two check. Optional FLAML and Keras/TensorFlow programs were researched and syntax checked, **not installed or executed**. Their results must remain unreported until actually run with recorded compatible versions and the stated development-only boundaries.

## Files to retain and what they mean

- `lesson.md`: full learner prose, programs, examples, changed practice, solutions and references.
- `visual-specifications.md`: complete inline figure and investigation contracts.
- `design.md`: source conservation, canonical coverage, claims/research record, sequence ownership and author review.
- `banknote-data.csv`: unchanged openly licensed input required for the offline teaching study.
- `data-source.json`: compact URL/hash/source-count record from ingestion.
- `author-calculations.py`: reproducible fixed observed study plus compact constructed calculations.
- `calculated-inputs.json`: actual folds, roles, predictions, warnings, outcomes and separately marked constructed fixtures.
- This provenance file: source, licensing, splitting and interpretation record.

These are necessary pending content artifacts, not disposable scratch. No downloaded ZIP, book PDF, model checkpoints, screenshots or temporary output images are retained. At implementation, derive a compact topic-local runtime asset from the documented evidence, preserve provenance, and follow the current retention policy for any temporary checks. Do not delete the source/calculation evidence merely because the page has been rendered once.
