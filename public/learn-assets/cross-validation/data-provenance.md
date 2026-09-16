# Cross-validation: input and quantitative provenance

The included `penguins.csv` is byte-identical to the preceding Feature Scaling packet's public Palmer Penguins input. Source https://raw.githubusercontent.com/allisonhorst/palmerpenguins/main/inst/extdata/penguins.csv; project https://allisonhorst.github.io/palmerpenguins/ declares CC0. Credit Allison Horst, Alison Hill and Kristen Gorman, and the original Palmer Station measurements by Gorman and colleagues. The original measurement context differs from our instructional species-classification task.

SHA-256 `f204db2c753b0937caac3cb35258562c14f073e4bbc76be24b4c51ce22767a93`; 15,241 bytes; all344 original rows in original order. Four numeric measurement features (bill length/depth and flipper length inmm, body mass ing) plus recorded sex. Species is the target; island,year and zero-based source-row position are excluded from X. Numeric missing cells and eleven missing sex entries remain in the source and are imputed only within the relevant training fold.

Three outer stratified folds use random_state41, sizes115/115/114; inner stratified folds use random_state73 inside each outer training subset. Candidate enumeration is neighbor count3,5,11 crossed with StandardScaler,RobustScaler, as shown in calculated-inputs.json. Exact ties take first enumerated candidate. The entire imputer/scaler/encoder/KNN pipeline is fitted in every inner fold. Outer training sizes are229/229/230. All original-row inner/outer indices, candidate scores, outer predictions/truth and selected settings are retained.

Recorded outer selections:11/Standard,3/Standard,3/Robust; correct114/115,114/115,112/114. Majority baselines51/115,51/115,50/114. Pooled340/344=.9883720930; unweighted fold mean.9883549453. Final all-data search selects3/Standard, inner selection score.9913043478. This familiar-data example demonstrates a reproducible protocol; it is not new independent evidence validating the preceding lesson's chosen scaler. The random-row split does not certify future-season or unseen-site deployment.

`author-calculations.py` performed this bounded calculation with read-only shared Python3.12.14, NumPy2.3.5, pandas3.0.1 and scikit-learn1.9.1. All actual fits were small and serial, with error_score='raise'; no warnings/failures appeared. The retained result contains no timing benchmark. Main manuscript program is a complete equivalent extraction, not claimed independently executed verbatim. Formal native/browser validation is deferred.

Constructed evidence in the same result file:

- Seven-row remainder-safe3-fold1NN trace, each exact held-out prediction and correct count.
- Exhaustive16-label-pattern selection calculation: two constant candidates average selected score.6875; all16 candidates score1; true independent fair-label future accuracy.5.
- Analytic random-hit probabilities for stated masses/trial counts.
- Sixteen-row two-by-two nested1NN/3NN example and actual recomputed label-edit traces for row7 and row3. The row3 case was selected after enumerating single-label edits to locate a clear instructional change in selected count; it is a constructed mechanism example, not a favorable empirical model comparison. Base and full changed states are preserved.

The tiny nested calculations were added in a bounded separate call using the same saved Python function; the expensive real pipeline calculation was not repeated for those changes. The author script recreates both sections in one execution if needed later. A transient Python import cache was removed after preserving these results. No downloaded PDFs, screenshots, disposable logs or new environment is retained. Keep the CSV/result/calculation pair as required inputs to the pending implementation.
