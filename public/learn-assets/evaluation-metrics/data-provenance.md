# Evaluation metrics data and calculation provenance

The small inspection, residual and document-ranking tables are original constructed instructional examples. Their coordinates, labels, probabilities and relevance judgments are not measured classifier benchmarks. metrics-calculations.py computes them and records exact results in metric-fixtures.json, including independently changed practice cases and nulls. They may be reproduced with the lesson. No external figure was copied.

The real experiment uses **Banknote Authentication**, Volker Lohweg (2012), [UCI record](https://archive.ics.uci.edu/dataset/267/banknote+authentication), DOI [10.24432/C55P57](https://doi.org/10.24432/C55P57), licensed [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Credit the creator, link the source and license, and disclose the subset/split/feature selection below. The source describes wavelet-transformed image features from banknote specimens. It supplies 1,372 rows, four continuous features and a class code. We retain the source's feature spelling `curtosis`; class1 is designated positive without inventing a genuine/forged code mapping.

Retrieved September 12, 2026 from https://archive.ics.uci.edu/static/public/267/banknote+authentication.zip; the archive was read in memory, and member `data_banknote_authentication.txt` parsed. The complete upstream archive is not retained as unnecessary duplication.

| Artifact | SHA-256 |
|---|---|
| Retrieved ZIP | 1e2acd9a2085fadf3d8145c12d3d22af853320d52294a6590c2eaf75fdc05227 |
| Original text member | d0539aaed2139ba7a587b3e34fb345ce503ff7d5d33dbf9912d8e195ce425cb9 |
| Retained banknote-subset.csv | d28fa993ed459d2f706816395475af08eebd2f394be67f2ad42dd9b511fc6b5a |

Take `numpy.random.default_rng(23).permutation(1372)[:480]`. Preserve that order and the original one-based source-row identifier. Rows0–319 are labeled `pool` in the shared CSV, rows320–399 `dev`, rows400–479 `test`. In this lesson **all320 pool rows are labeled training data**; this differs deliberately from the semi-supervised and active-learning protocols that reuse the same retained subset. It is not a direct label-budget comparison between lessons. All480 source IDs are unique and the split sets are disjoint.

banknote-evaluation.py uses only variance and entropy. Fit a StandardScaler/LogisticRegression(C=1,max_iter=500) pipeline on the320 training rows; fit no transformation on development or test. Choose a threshold from distinct development scores plus no-alert option to minimize FP+5FN, with correct decisions cost0 and a highest-threshold tie break. Freeze the model and threshold before evaluating test predictions. Compare a fixed.5 threshold and constant training-prior score. The costs are constructed decision units, not measured monetary consequences. The extra constant-prior action at threshold1/6 was added during author checking to make the cost baseline informative; it is analytically derived from the fixed cost matrix, not tuned against test labels. Do not claim every displayed baseline was frozen before the author's first draft run.

The selected threshold .12013000807724628 costs18 on development but26 on test; fixed.5 costs24 on test. This inconvenient test comparison was retained. No subsequent test-optimal threshold replaces it. Raw development/test probabilities, labels, selected threshold, all candidate development costs, source IDs, curve arrays, counts and aggregate scores are retained in banknote-evaluation-results.json. Those curves are **measured** from this fitted model/split; they are not estimates of universal model performance or population confidence bands.

Both programs were executed with Python3.12.14, NumPy2.3.5 and scikit-learn1.9.1 on September12,2026. The metric program's grouped AUC/AP and transformed-gain NDCG agree with library calculations; equal-score permutation, threshold-interval and equal-grade-swap nulls are asserted. The regression unit conversion, ranking edit, independent practice arithmetic and streaming example are recorded. These are bounded author checks for content accuracy. Production translation, browser interaction, keyboard/screen-reader behavior, responsive graphics, route integration and independent review remain phase-two work.

Retain all files in this packet until implementation consumes them. The small CSV is duplicated locally so this pending packet runs independently of another draft; production should choose a deliberate shared data-download owner if appropriate. Do not load every lesson's data eagerly or delete pending provenance as scratch.
