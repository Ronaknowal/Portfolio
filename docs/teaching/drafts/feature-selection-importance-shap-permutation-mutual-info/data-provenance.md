# Feature-selection input and numeric evidence

Research/write packet, 12 September 2026. The data are an observed historical laboratory collection; the count, XOR, duplicate-sensor and polynomial games are separately declared constructed fixtures. No constructed plot is labeled as a native benchmark or measured population finding.

## Observed input

- Source: Aeberhard, S. and Forina, M. (1992), **Wine**, UCI Machine Learning Repository, [DOI 10.24432/C5PC7J](https://doi.org/10.24432/C5PC7J), [source archive](https://archive.ics.uci.edu/static/public/109/wine.zip).
- License: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Credit the source and identify subsequent split/analysis as this lesson's work.
- Retrieved 2026-09-12. ZIP SHA-256 `2bae62c4481220623579d4c4fb36b55652b6b75e06e49fa1981b8198362dfdab`. The archive was read in memory and is not retained.
- Retained unchanged numeric member `wine.data`, 10,782 bytes, SHA-256 `6be6b1203f3d51df0b553a70e57b8a723cd405683958204f96d23d7cd6aea659`. All178 rows, first column cultivar class1/2/3, followed by thirteen measurement fields. No header, missing values, downsampling, synthesized observations or altered source order.
- The complete source `wine.names` was inspected. Source measurements concern three cultivars grown in the same region of Italy; they are not quality ratings. The supplied field table/names do not establish physical units for all fields. Retain raw source scales without guessed units. No independent vineyard, vintage, run or laboratory group IDs are provided for evaluating those transfer claims.
- `data-source.json` stores compact machine-readable attribution and hashes. The upstream names text and ZIP are not necessary retained copies. UCI's canonical Wine page returned a retrieval error during research; the DOI, actual archive and the repository's body for numeric dataset109 identified the matching Wine description and license. A slug mismatch in an indexed numeric-ID page was not treated as a different dataset or cited as an unrelated study.

## Declared study and execution

`author-calculations.py` ran in the existing read-only lesson runtime: Python3.12.14, NumPy2.3.5, scikit-learn1.9.1. It writes `calculated-inputs.json`. This is a bounded content-evidence calculation, not publication QA or a full benchmark campaign. Exactly eleven fits: nine inner CV candidates, one chosen six-feature refit, one separately predeclared four-feature refit. No installation or shared runtime change.

The outer138/40 stratified boundary uses seed51; fit100/inspection38 within development uses seed52. Three inner stratified folds seed53 compare k3,6,13. `SelectKBest` uses MI, continuous input flag, n_neighbors3 and seed54. Each candidate fits its selector solely on its own training rows. A depth3 tree, minimum leaf5, seed55 is the predictor. Equal size scores prefer the earlier smaller listed k. The separate four-field tree uses raw indices0,1,6,12 and the same settings. It is declared for tractable exhaustive explanation, not selected by its inspection result. Forty reserve rows are identified but never predicted/scored. The row-level result does not estimate a new region or experimental process.

The full evidence records split IDs, fold fitting/validation IDs, feature masks, MI estimates in nats, actual fold predictions, final inspection predictions, counts, confusion matrix and saved tree. PI uses exactly twenty seeded donor permutations, seed56+r, shared across feature comparisons, with accuracy decrease as the quantity and population SD of those twenty repeats. Native MDI uses its distinct normalized training-impurity quantity. Counts are9 inner fits+2 refits=11, not repeated for a favorable score.

Both models give36/38 inspection accuracy; the training-majority class2 baseline gives15/38. These are preserved non-winning outcomes. The six-feature identities vary across inner folds. No significance or universality is inferred from this small collection.

## Attribution record and bounded follow-up

The explanation target is class1 probability of the four-field tree, not its raw margin, actual target label or quality. The principal reference is all100 fitting rows with equal weights. Each of sixteen coalitions overwrites retained instance coordinates in those actual donor rows; exact subset weights allocate the average outputs. Twelve predeclared inspection rows are explained, beginning at zero-based source row104. Values, source measurements, classes, baselines, phi, reconstruction errors and full saved tree are retained.

An initial exploratory implementation used the first12 sorted fitting IDs. Source order groups classes, so that was an unsuitable hidden approximation to the fitting population. Before manuscript freeze it was explicitly replaced by the entire100-row reference, giving the meaningful baseline0.33. This changed the explanation question, not the fitted model, feature selection, inspection accuracy or assessment boundary. The existing saved tree was used for this bounded recalculation; no refit was needed. Its hard predictions matched native output on all38 inspection cases. The final author program now regenerates the same full-reference calculation directly.

The final12 sorted fitting IDs are retained only as an explicitly labeled class3-cohort **contrast**, not as a representative sample. Its baseline0 and changed signed contributions are accompanied by the same instance output0. Input contrast alcohol12.51→13.5 gives output1; null edit malic1.73→4.1 preserves every coalition and contribution. Saved-tree inference casts raw inputs to float32 before threshold comparison, matching sklearn's input convention. The author program contains both changed cases for reproducibility. Phase two must also compare tree probability outputs and threshold-near inputs with native inference.

No `shap` package was installed or executed. Its optional manuscript comparison is explicitly unexecuted and remains a phase-two check. All current numeric explanations come from the complete declared exhaustive game and saved/native tree evaluations, not fabricated SHAP-library output. The displayed educational programs were assembled from these checked computations; their independent verbatim execution, browser rendering, mobile/accessibility and loading tests are still open.

## Retention

Keep the eight topic-owned packet files: full lesson, specifications, design, this provenance, source metadata, unchanged Wine input, compact author calculation and resulting numeric record. They are necessary for the pending implementation, attribution and numerical oracle. There are no generated screenshots, downloaded papers, cache folders, full ZIP copies or disposable logs to preserve. Root owns hashes for checkpointed manuscript/specification artifacts; do not copy these outputs into a global eagerly loaded registry.
